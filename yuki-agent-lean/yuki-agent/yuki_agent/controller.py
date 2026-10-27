"""Agent controller: owns the request lifecycle (validate -> classify -> tools -> answer -> persist)."""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

from . import intent as intent_mod
from .citations import CitationManager
from .context import chat_messages, qa_messages, extract_json, build_messages
from .conversation import ConversationManager
from .errors import (AgentTimeout, Cancelled, PermissionDenied, ProviderError, ProviderUnavailable,
                     VectorStoreError, YukiError)
from .formatter import ActivityLog, AgentResponse
from .lang import pick_language, t
from .safety import validate_message
from .tool_impls import build_default_registry
from .tools import ToolContext, ToolExecutor

log = logging.getLogger("yuki.agent")

LABELS = {
    "search_documents": "Searching selected documents", "summarize_document": "Summarizing document",
    "explain_concept": "Explaining concept", "generate_flashcards": "Generating flashcards",
    "generate_quiz": "Generating quiz", "compare_documents": "Comparing documents",
    "generate_research_ideas": "Generating research ideas", "analyze_knowledge_gaps": "Analyzing knowledge gaps",
    "create_study_plan": "Creating study plan", "save_research_note": "Saving result",
}


class ChatRequest:
    def __init__(self, message, session_id=None, selected_document_ids=None, project_id=None,
                 preferences=None, regenerate=False):
        self.message, self.session_id = message, session_id
        # None = not provided (reuse the session's selection); [] = the user explicitly selected nothing
        self.selected_document_ids = None if selected_document_ids is None else list(selected_document_ids)
        self.project_id, self.preferences, self.regenerate = project_id, preferences or {}, regenerate


class AgentController:
    def __init__(self, settings, db, retriever, llm, registry=None):
        self.settings, self.db, self.retriever, self.llm = settings, db, retriever, llm
        self.registry = registry or build_default_registry()
        self.conversation = ConversationManager(db, settings)

    # ------------------------------------------------------------ public API
    def handle(self, req: ChatRequest, user_id, on_event=None, cancel=None) -> AgentResponse:
        t0 = time.monotonic()
        resp = AgentResponse()
        activity = ActivityLog(on_event)
        session = None
        try:
            message = validate_message(req.message, self.settings.max_message_chars)
            session = self.conversation.get_or_create(user_id, req.session_id, message, req.project_id)
            resp.session_id = session["id"]
            if req.regenerate:
                prev = self.db.delete_last_exchange(user_id, session["id"])
                message = prev or message
            project_id = req.project_id or session.get("project_id")
            if project_id and not self.db.get_project(user_id, project_id):
                raise PermissionDenied("Project not found.")
            docs = self._resolve_documents(user_id, session, req)
            if req.preferences:
                self.db.update_session(user_id, session["id"], preferences=req.preferences)
            prefs = req.preferences or session.get("preferences") or {}
            history = self.conversation.history(user_id, session["id"])
            self.conversation.add(session["id"], "user", message)

            lang, forced = pick_language(prefs, message)
            resp.metadata["language"] = lang
            ctx = ToolContext(user_id, session["id"], self.settings, self.db, self.retriever, self.llm,
                              CitationManager(), project_id, t0 + self.settings.request_timeout_s, False, cancel,
                              lang, forced)
            executor = ToolExecutor(self.registry, self.settings, ctx)
            self._dispatch(message, docs, session, project_id, history, prefs, ctx, executor, activity, resp)
            resp.metadata["tool_calls"] = executor.calls
        except Cancelled:
            resp.status, resp.error = "cancelled", "Generation was cancelled."
        except AgentTimeout as e:
            resp.status, resp.error = "error", str(e)
        except ProviderUnavailable as e:
            resp.status = "error"
            resp.error = t("model_down", resp.metadata.get("language", "en"), detail=e)
        except ProviderError as e:
            resp.status, resp.error = "error", f"The language model reported an error: {e}"
        except VectorStoreError as e:
            resp.status, resp.error = "error", f"The document index is unavailable: {e}"
        except YukiError as e:
            resp.status, resp.error = "error", str(e)
        except Exception as e:   # never log message/document content
            log.error("unexpected agent failure: %s", type(e).__name__)
            resp.status, resp.error = "error", "Something went wrong while processing your request."

        if resp.status in ("error", "cancelled") and not resp.answer:
            resp.answer = resp.error or ""
        resp.tool_activity = activity.steps
        for s in resp.tool_activity:
            if s["status"] == "running":
                s["status"] = "failed" if resp.status == "error" else "cancelled"
        resp.metadata.update(provider=getattr(self.llm, "name", "?"), model=getattr(self.llm, "model", "?"),
                             elapsed_ms=int((time.monotonic() - t0) * 1000))
        if session is not None and resp.answer:
            self.conversation.add(session["id"], "assistant", resp.answer, {
                "status": resp.status, "error": resp.status in ("error", "cancelled"),
                "saveable": resp.status == "completed", "sources": resp.sources,
                "warnings": resp.warnings})
        return resp

    def resolve_pending(self, user_id, pending_id, approve, on_event=None):
        """User approved/rejected a confirmation-gated action (e.g. saving a note)."""
        resp, activity = AgentResponse(), ActivityLog(on_event)
        pend = self.db.get_pending(user_id, pending_id)
        if not pend or pend["status"] != "pending":
            raise PermissionDenied("That action was not found or has already been handled.")
        resp.session_id = pend["session_id"] or ""
        age = datetime.now(timezone.utc) - datetime.fromisoformat(pend["created_at"])
        if age.total_seconds() > 24 * 3600:
            self.db.set_pending_status(pending_id, "expired")
            raise PermissionDenied("That confirmation expired. Please ask me again.")
        if not approve:
            self.db.set_pending_status(pending_id, "rejected")
            sess = self.db.get_session(user_id, pend["session_id"]) if pend["session_id"] else None
            resp.answer = t("nothing_saved", ((sess or {}).get("preferences") or {}).get("language", "en"))
            return resp
        ctx = ToolContext(user_id, pend["session_id"], self.settings, self.db, self.retriever, self.llm,
                          CitationManager(), None, time.monotonic() + 30, True, None)
        executor = ToolExecutor(self.registry, self.settings, ctx)
        step = activity.start(LABELS["save_research_note"], pend["tool"])
        res = executor.execute(pend["tool"], pend["arguments"])
        if res.ok:
            self.db.set_pending_status(pending_id, "executed")
            activity.finish(step, "done", res.data.get("title"), res.duration_ms)
            resp.answer, resp.structured = res.text or "Saved.", {"type": "note_saved", **res.data}
        else:
            self.db.set_pending_status(pending_id, "failed")
            activity.finish(step, "failed", res.error, res.duration_ms)
            resp.status, resp.error, resp.answer = "error", res.error, f"Nothing was saved. {res.error}"
        resp.tool_activity = activity.steps
        if pend["session_id"]:
            self.conversation.add(pend["session_id"], "assistant", resp.answer,
                                  {"status": resp.status, "error": resp.status == "error", "saveable": False})
        return resp

    # ------------------------------------------------------------ internals
    def _resolve_documents(self, user_id, session, req):
        if req.selected_document_ids is not None:          # explicit choice from the user: strict check
            ids = list(dict.fromkeys(req.selected_document_ids))[:20]
            found = self.db.get_documents(user_id, ids)
            if len(found) != len(ids) or any(d["status"] != "ready" for d in found):
                raise PermissionDenied("One or more selected documents were not found or are not ready.")
            self.db.update_session(user_id, session["id"], selected_docs=ids)
            return ids
        remembered = list(session.get("selected_docs") or [])[:20]   # remembered choice: drop documents deleted since
        if not remembered:
            return []
        ok = {d["id"] for d in self.db.get_documents(user_id, remembered) if d["status"] == "ready"}
        kept = [i for i in remembered if i in ok]
        if kept != remembered:
            self.db.update_session(user_id, session["id"], selected_docs=kept)
        return kept

    def _run(self, executor, activity, tool, args, label=None):
        step = activity.start(label or LABELS.get(tool, tool), tool)
        try:
            res = executor.execute(tool, args)
        except Exception:
            activity.finish(step, "failed")
            raise
        status = {"ok": "done", "needs_confirmation": "waiting"}.get(res.status, "failed")
        detail = res.error or ("duplicate call reused" if res.duplicate else None)
        activity.finish(step, status, detail, res.duration_ms)
        return res

    def _apply_tool_result(self, res, ctx, resp, only_cited=False):
        resp.warnings.extend(w for w in res.warnings if w not in resp.warnings)
        if res.status == "needs_confirmation":
            resp.status, resp.pending_action = "needs_confirmation", res.pending_action
            resp.answer = t("confirm_save", ctx.language) + "\n\n" + res.pending_action["preview"]
            return
        if res.status in ("error", "insufficient_evidence"):
            resp.status = "insufficient_evidence" if res.status == "insufficient_evidence" else "error"
            resp.error = res.error
            resp.answer = (res.error if resp.status == "error"
                           else (f"{res.error} " if ctx.language == "en" else "") + t("insufficient", ctx.language))
            return
        text, sources, dropped = ctx.citations.finalize(res.text or "", res.cited_labels, only_cited)
        if dropped:
            resp.warnings.append("Some citation markers did not match retrieved sources and were removed.")
        resp.answer, resp.sources = text, sources
        if res.grounded is False:
            resp.metadata["grounded"] = False
        data = {k: v for k, v in res.data.items()}
        if data:
            resp.structured = {"type": res.tool, **data}

    def _dispatch(self, message, docs, session, project_id, history, prefs, ctx, executor, activity, resp):
        has_docs = bool(self.db.list_documents(ctx.user_id, ready_only=True))
        step = activity.start("Understanding request")
        it = intent_mod.classify(message, has_documents=has_docs)
        routed = None
        if self.settings.llm_routing and it.confidence < 0.6 and has_docs:
            routed = intent_mod.llm_route(ctx, message, self.registry, docs)
        activity.finish(step, "done", f"Intent: {it.name}" if not routed else f"Model chose: {routed[0]}", 0)
        resp.metadata["intent"] = it.name

        if routed:
            tool, args = routed
            res = self._run(executor, activity, tool, args)
            return self._apply_tool_result(res, ctx, resp)
        if it.name == "chat":
            return self._plain_chat(message, history, prefs, ctx, activity, resp)
        if it.name == "qa":
            return self._answer_question(message, docs, history, prefs, ctx, executor, activity, resp)

        built = self._build_args(it, message, docs, session, project_id, ctx, resp)
        if built is None:         # needs_input already set on resp
            return
        tool, args = built
        res = self._run(executor, activity, tool, args)
        self._apply_tool_result(res, ctx, resp, only_cited=(tool == "summarize_document"))

    def _plain_chat(self, message, history, prefs, ctx, activity, resp):
        step = activity.start("Generating answer")
        ts = time.monotonic()
        resp.answer = ctx.llm_chat(chat_messages(message, history, prefs), temperature=0.4).strip()
        activity.finish(step, "done", "No documents used", int((time.monotonic() - ts) * 1000))
        resp.metadata["grounded"] = False

    def _contextualize(self, message, history):
        """Short follow-ups ('what about its limits?') borrow the previous user question for retrieval."""
        if len(message.split()) < 7:
            prev = [m["content"] for m in history if m["role"] == "user"]
            if prev:
                return f"{prev[-1]} {message}"[:1000]
        return message

    def _answer_question(self, message, docs, history, prefs, ctx, executor, activity, resp):
        label = "Searching selected documents" if docs else "Searching your documents"
        res = self._run(executor, activity, "search_documents",
                        {"query": self._contextualize(message, history), "selected_document_ids": docs,
                         "top_k": max(1, min(self.settings.top_k, 20))}, label)
        if not res.ok:
            return self._apply_tool_result(res, ctx, resp)
        resp.warnings.extend(res.warnings)
        results = res.data["results"]
        if not results:
            resp.status, resp.answer = "insufficient_evidence", t("insufficient", ctx.language)
            activity.note("Reviewing retrieved evidence", "No passages passed the relevance threshold")
            return
        activity.note("Reviewing retrieved evidence",
                      f"{len(results)} passage(s) from {len({r['document_id'] for r in results})} document(s)")
        evidence = ctx.citations.evidence_for_labels([r["label"] for r in results])
        step = activity.start("Generating answer")
        ts = time.monotonic()
        raw = ctx.llm_chat(qa_messages(message, evidence, history, prefs), temperature=0.2)
        activity.finish(step, "done", None, int((time.monotonic() - ts) * 1000))
        if "INSUFFICIENT_EVIDENCE" in raw:
            resp.status, resp.answer = "insufficient_evidence", t("insufficient", ctx.language)
            return
        text, sources, dropped = ctx.citations.finalize(raw)
        if dropped:
            resp.warnings.append("Some citation markers did not match retrieved sources and were removed.")
        if not any(s["cited"] for s in sources):
            resp.warnings.append("The answer does not explicitly cite the retrieved passages; verify it against the sources below.")
        bad = ctx.citations.unverified_quotes(text)
        if bad:
            resp.warnings.append("The answer contains quoted text that was not found in the retrieved passages: "
                                 + "; ".join(f'"{b}..."' for b in bad[:3]))
        resp.answer, resp.sources = text.strip(), sources

    # -- argument building per intent
    def _need(self, resp, text):
        resp.status, resp.answer = "needs_input", text
        return None

    def _build_args(self, it, message, docs, session, project_id, ctx, resp):
        p, name = it.params, it.name
        topic = intent_mod.extract_topic(message)
        if name == "summarize":
            target = docs[:1] if len(docs) == 1 else []
            if not target:
                ready = self.db.list_documents(ctx.user_id, ready_only=True)
                if len(docs) == 0 and len(ready) == 1:
                    target = [ready[0]["id"]]
            if not target and not self.db.list_documents(ctx.user_id, ready_only=True):
                # nothing uploaded: "give me an overview of X" is a general-knowledge request, not a document summary
                return "explain_concept", {"topic": message, "difficulty_level": p.get("level", "intermediate"),
                                           "optional_document_ids": []}
            if not target:
                names = ", ".join(d["filename"] for d in self.db.list_documents(ctx.user_id, True)[:8]) or "none uploaded"
                return self._need(resp, f"Which document should I summarise? Select exactly one. Available: {names}.")
            return "summarize_document", {"document_id": target[0], "summary_length": p.get("length", "medium"),
                                           "focus": p.get("focus", "")}
        if name == "explain":
            return "explain_concept", {"topic": message, "difficulty_level": p.get("level", "intermediate"),
                                       "optional_document_ids": docs}
        if name == "flashcards":
            return "generate_flashcards", {"topic": topic, "selected_document_ids": docs, "count": p.get("count", 10)}
        if name == "quiz":
            return "generate_quiz", {"topic": topic, "selected_document_ids": docs,
                                     "difficulty": p.get("difficulty", "medium"), "question_count": p.get("count", 10)}
        if name == "compare":
            if len(docs) < 2:
                return self._need(resp, "Select at least two documents to compare, then ask again.")
            return "compare_documents", {"document_ids": docs, "comparison_topic": topic if topic != "key concepts and main ideas" else "overall"}
        if name == "ideas":
            return "generate_research_ideas", {"research_topic": message, "selected_document_ids": docs}
        if name == "gaps":
            return "analyze_knowledge_gaps", {"topic": topic, "selected_document_ids": docs}
        if name == "study_plan":
            return self._study_plan_args(message, ctx, resp)
        if name == "save_note":
            return self._save_note_args(p, ctx, project_id, resp)
        return self._need(resp, "I'm not sure what you'd like me to do. Could you rephrase?")

    def _study_plan_args(self, message, ctx, resp):
        prompt = ("Extract study-plan details from the request. Use null for anything not stated; never guess.\n"
                  f"Request: {message}\nReply ONLY with JSON: "
                  '{"subjects":["..."],"available_study_time":"...","deadline":"...","learning_goals":"..."}')
        try:
            d = extract_json(ctx.llm_chat(build_messages(prompt), json_mode=True, temperature=0.0))
        except ValueError:
            d = {}
        d = d if isinstance(d, dict) else {}
        missing = [k for k in ("subjects", "available_study_time", "deadline") if not d.get(k)]
        if missing:
            nice = {"subjects": "subjects", "available_study_time": "how much time you have to study",
                    "deadline": "your deadline"}
            return self._need(resp, "To build a realistic plan I still need: " + ", ".join(nice[m] for m in missing) + ".")
        return "create_study_plan", {"subjects": d["subjects"], "available_study_time": str(d["available_study_time"]),
                                     "deadline": str(d["deadline"]), "learning_goals": str(d.get("learning_goals") or "")}

    def _save_note_args(self, params, ctx, project_id, resp):
        last = self.conversation.last_assistant(ctx.user_id, ctx.session_id)
        if not last:
            return self._need(resp, "There's nothing to save yet. Ask me something first, then say 'save this'.")
        if not project_id:
            projects = self.db.list_projects(ctx.user_id)
            if len(projects) == 1:
                project_id = projects[0]["id"]
            elif not projects:
                return self._need(resp, "You don't have a project yet. Create one first, then ask me to save again.")
            else:
                names = "; ".join(f"{p['name']} (id {p['id']})" for p in projects[:10])
                return self._need(resp, f"Which project should I save it to? {names}")
        content = last["content"]
        first = next((l for l in content.splitlines() if l.strip()), "Note")
        title = params.get("title") or re.sub(r"[#*_>`]", "", first).strip()[:80] or "Note"
        refs = [f"{s['filename']}" + (f", p.{s['page']}" if s.get("page") else "")
                for s in last["metadata"].get("sources", []) if s.get("cited")]
        return "save_research_note", {"project_id": project_id, "title": title, "content": content,
                                      "source_references": list(dict.fromkeys(refs))}
