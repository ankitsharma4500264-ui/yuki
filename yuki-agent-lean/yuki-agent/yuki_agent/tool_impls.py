"""The ten tools. Handlers do the work; the executor validates, authorises and limits them."""
from __future__ import annotations

import re
import concurrent.futures
from datetime import date

from .context import build_messages, extract_json
from .errors import InsufficientEvidence, ToolFailure
from .rag import Hit
from .safety import detect_injection
from .tools import ArgSpec, ToolOutput, ToolRegistry, ToolSpec

INJECTION_WARNING = ("Some retrieved passages contain text that looks like instructions. It was treated as "
                     "document content and ignored.")
GENERAL_WARNING = "Not grounded in your documents: this uses general knowledge."


# ------------------------------------------------------------------ helpers
def _retrieve(ctx, query, ids, top_k=None):
    ctx.check()
    return ctx.retriever.search(query, ctx.user_id, ids or None, top_k)


def _evidence(ctx, hits, warnings):
    if any(detect_injection(h.record.text) for h in hits) and INJECTION_WARNING not in warnings:
        warnings.append(INJECTION_WARNING)
    return ctx.citations.evidence_block(hits)


def _study_evidence(ctx, topic, ids, warnings, top_k):
    """Evidence for study tools. Selected docs with no hits -> insufficient; no selection -> general fallback."""
    hits = _retrieve(ctx, topic, ids, top_k)
    if hits:
        return hits, _evidence(ctx, hits, warnings), True
    if ids:
        raise InsufficientEvidence("The selected documents do not contain enough relevant material on this topic.")
    warnings.append(GENERAL_WARNING)
    return [], "", False


def _label_ok(ctx, v):
    return v if isinstance(v, str) and ctx.citations.has(v.strip()) else None


def _collect(ctx, make_messages, list_key, parse, want, key, attempts=2):
    items, seen = [], set()
    for _ in range(attempts):
        need = want - len(items)
        if need <= 0:
            break
        raw = ctx.llm_chat(make_messages(need, [key(i) for i in items]), json_mode=True, temperature=0.4)
        try:
            obj = extract_json(raw)
        except ValueError:
            continue
        for it in (obj.get(list_key) if isinstance(obj, dict) else None) or []:
            try:
                parsed = parse(it)
            except (KeyError, TypeError, ValueError, AttributeError):
                parsed = None
            if parsed and key(parsed).lower() not in seen:
                seen.add(key(parsed).lower())
                items.append(parsed)
    if not items:
        raise ToolFailure("The model did not return valid structured output. Try again or use a larger model.")
    return items[:want]


def _txt(v):
    if not isinstance(v, str) or not v.strip():
        raise ValueError("empty")
    return v.strip()


# ------------------------------------------------------------------ A. search_documents
def search_documents(args, ctx):
    hits = _retrieve(ctx, args["query"], args["selected_document_ids"], args["top_k"])
    warnings = []
    _evidence(ctx, hits, warnings)
    results = []
    for h in hits:
        r = h.record
        results.append({"label": ctx.citations.register(h), "document_id": r.document_id, "filename": r.filename,
                        "page": r.page, "chunk_id": r.chunk_id, "text": r.text,
                        "score": None if h.score is None else round(h.score, 4)})
    return ToolOutput(data={"results": results}, warnings=warnings, grounded=bool(results))


# ------------------------------------------------------------------ B. summarize_document
LENGTHS = {"short": "about 5 bullet points (under 120 words)",
           "medium": "about 300 words",
           "long": "about 600 words with headed sections"}


def _batches(chunks, size):
    cur, n = [], 0
    for c in chunks:
        if cur and n + len(c.text) > size:
            yield cur
            cur, n = [], 0
        cur.append(c)
        n += len(c.text)
    if cur:
        yield cur


def summarize_document(args, ctx):
    doc = ctx.db.get_document(ctx.user_id, args["document_id"])
    chunks = ctx.retriever.document_chunks(args["document_id"], ctx.user_id)
    if not doc or not chunks:
        raise InsufficientEvidence("That document has no indexed text to summarise.")
    warnings, focus = [], args["focus"]
    focus_line = f" Focus on: {focus}." if focus else ""
    groups = list(_batches(chunks, ctx.settings.summary_section_chars))
    cap = ctx.settings.max_summary_sections
    if len(groups) > cap:
        step = len(groups) / cap
        groups = [groups[int(i * step)] for i in range(cap)]
        warnings.append(f"Long document: sampled {cap} evenly spaced sections instead of reading every one.")
    blocks = [_evidence(ctx, [Hit(c, None) for c in g], warnings) for g in groups]   # labels assigned in order

    def summarise(block):
        return ctx.llm_chat(build_messages(f"{block}\n\nSummarise this section of '{doc['filename']}' in 4-6 "
                                           f"concise bullet points, citing labels.{focus_line}"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:      # sections are independent
        partials = list(pool.map(summarise, blocks))
    while len(partials) > 8:   # hierarchical reduce for very long documents
        partials = [ctx.llm_chat(build_messages(
            "Merge these partial summaries into one coherent summary, keeping the [S#] labels."
            f"{focus_line}\n\n" + "\n\n".join(partials[i:i + 8]))) for i in range(0, len(partials), 8)]
    body = "\n\n".join(partials)
    final = ctx.llm_chat(build_messages(
        f"Write the final summary of '{doc['filename']}' as {LENGTHS[args['summary_length']]}.{focus_line} "
        f"Keep [S#] labels where claims come from the text.\n\nPartial summaries:\n{body}"))
    return ToolOutput(data={"document_id": doc["id"], "filename": doc["filename"], "sections": len(groups)},
                      text=final, warnings=warnings, grounded=True)


# ------------------------------------------------------------------ C. explain_concept
LEVELS = {"beginner": "plain, simple English, short sentences, everyday analogies, no jargon",
          "intermediate": "clear explanation assuming basic background knowledge",
          "advanced": "precise, technical explanation with nuances and edge cases"}


def explain_concept(args, ctx):
    ids, warnings = args["optional_document_ids"], []
    hits = _retrieve(ctx, args["topic"], ids, ctx.settings.top_k) if (ids or ctx.db.list_documents(ctx.user_id, True)) else []
    level = LEVELS[args["difficulty_level"]]
    if hits:
        block = _evidence(ctx, hits, warnings)
        prompt = (f"{block}\n\nExplain: {args['topic']}\nStyle: {level}.\n"
                  "Use the evidence first and cite labels. Put anything NOT supported by the evidence under a final "
                  "heading 'General knowledge (not from your documents)'. End with a 2-sentence 'In short:' summary.")
        grounded = True
    else:
        if ids:
            warnings.append("No relevant passages were found in the selected documents; this is general knowledge.")
        else:
            warnings.append(GENERAL_WARNING)
        prompt = (f"Explain: {args['topic']}\nStyle: {level}.\nThis is general knowledge, not from the user's "
                  "documents; say so briefly. End with a 2-sentence 'In short:' summary.")
        grounded = False
    return ToolOutput(text=ctx.llm_chat(build_messages(prompt)), warnings=warnings, grounded=grounded)


# ------------------------------------------------------------------ D. generate_flashcards
def generate_flashcards(args, ctx):
    warnings, want = [], args["count"]
    _, block, grounded = _study_evidence(ctx, args["topic"], args["selected_document_ids"], warnings,
                                         max(8, min(want, 15)))

    def make(need, existing):
        avoid = f" Do not repeat these questions: {existing}." if existing else ""
        return build_messages(
            f"{block}\n\nCreate {need} distinct flashcards on: {args['topic']}.{avoid}\n"
            'Return ONLY JSON: {"flashcards":[{"question":"...","answer":"...","source":"S1 or null"}]}. '
            "source must be an evidence label that supports the answer, or null if from general knowledge.")

    def parse(it):
        return {"question": _txt(it["question"]), "answer": _txt(it["answer"]), "source": _label_ok(ctx, it.get("source"))}

    cards = _collect(ctx, make, "flashcards", parse, want, lambda c: c["question"])
    if len(cards) < want:
        warnings.append(f"Generated {len(cards)} of the {want} requested flashcards.")
    lines = []
    for i, c in enumerate(cards, 1):
        src = f" [{c['source']}]" if c["source"] else ""
        lines.append(f"**{i}. {c['question']}**\n{c['answer']}{src}")
    return ToolOutput(data={"flashcards": cards}, text="\n\n".join(lines), warnings=warnings, grounded=grounded,
                      cited_labels=[c["source"] for c in cards if c["source"]])


# ------------------------------------------------------------------ E. generate_quiz
def generate_quiz(args, ctx):
    warnings, want = [], args["question_count"]
    _, block, grounded = _study_evidence(ctx, args["topic"], args["selected_document_ids"], warnings,
                                         max(8, min(want, 15)))

    def make(need, existing):
        avoid = f" Do not repeat: {existing}." if existing else ""
        return build_messages(
            f"{block}\n\nWrite {need} {args['difficulty']} quiz questions on: {args['topic']}. About 70% multiple-choice "
            f"(exactly 4 options) and the rest short-answer.{avoid}\nReturn ONLY JSON: "
            '{"questions":[{"type":"mcq","question":"...","options":["A","B","C","D"],"answer_index":0,'
            '"explanation":"...","source":"S1 or null"},{"type":"short","question":"...","answer":"...",'
            '"explanation":"...","source":"S1 or null"}]}. answer_index is 0-3.')

    def parse(it):
        base = {"question": _txt(it["question"]), "explanation": _txt(it.get("explanation") or "See source."),
                "source": _label_ok(ctx, it.get("source"))}
        if it.get("type") == "mcq":
            opts = [_txt(o) for o in it["options"]]
            idx = it["answer_index"]
            if len(opts) != 4 or len(set(o.lower() for o in opts)) != 4 or not isinstance(idx, int) or not 0 <= idx <= 3:
                return None
            return {**base, "type": "mcq", "options": opts, "answer_index": idx}
        return {**base, "type": "short", "answer": _txt(it["answer"])}

    qs = _collect(ctx, make, "questions", parse, want, lambda q: q["question"])
    if len(qs) < want:
        warnings.append(f"Generated {len(qs)} of the {want} requested questions.")
    out, key = [], []
    for i, q in enumerate(qs, 1):
        src = f" [{q['source']}]" if q["source"] else ""
        if q["type"] == "mcq":
            opts = "\n".join(f"   {'ABCD'[j]}. {o}" for j, o in enumerate(q["options"]))
            out.append(f"**Q{i}. {q['question']}**\n{opts}")
            key.append(f"Q{i}: {'ABCD'[q['answer_index']]} - {q['explanation']}{src}")
        else:
            out.append(f"**Q{i}. {q['question']}** (short answer)")
            key.append(f"Q{i}: {q['answer']} - {q['explanation']}{src}")
    text = "\n\n".join(out) + "\n\n---\n**Answer key**\n" + "\n".join(key)
    return ToolOutput(data={"questions": qs}, text=text, warnings=warnings, grounded=grounded,
                      cited_labels=[q["source"] for q in qs if q["source"]])


# ------------------------------------------------------------------ F. compare_documents
def compare_documents(args, ctx):
    warnings, blocks, names = [], [], []
    q = f"{args['comparison_topic']} objectives methodology findings limitations conclusions"
    for did in args["document_ids"]:
        hits = _retrieve(ctx, q, [did], ctx.settings.top_k)
        doc = ctx.db.get_document(ctx.user_id, did)
        name = doc["filename"] if doc else did
        if not hits:
            warnings.append(f"No relevant passages were found in '{name}'; it is excluded from the comparison.")
            continue
        names.append(name)
        blocks.append(f"Document: {name}\n{_evidence(ctx, hits, warnings)}")
    if len(blocks) < 2:
        raise InsufficientEvidence("Need relevant passages from at least two documents to compare them.")
    prompt = ("\n\n".join(blocks) + f"\n\nCompare these documents on: {args['comparison_topic']}.\n"
              "Sections: Objectives, Methodology, Findings, Limitations, Conclusions, Similarities, Differences. "
              "Cite labels for every claim about a document, and write 'Not stated in the retrieved passages' "
              "when something cannot be verified. Never mix up which document a claim comes from.")
    return ToolOutput(data={"documents": names}, text=ctx.llm_chat(build_messages(prompt)), warnings=warnings,
                      grounded=True)


# ------------------------------------------------------------------ G. generate_research_ideas
def generate_research_ideas(args, ctx):
    ids, warnings = args["selected_document_ids"], []
    hits = _retrieve(ctx, args["research_topic"], ids, ctx.settings.top_k) if ids else []
    block = _evidence(ctx, hits, warnings) + "\n\n" if hits else ""
    if ids and not hits:
        warnings.append("No relevant passages were found in the selected documents; ideas are not tied to them.")
    field_line = f" Field: {args['research_field']}." if args["research_field"] else ""
    rel = ("Relate ideas to the evidence with [S#] labels where relevant, without claiming full literature coverage."
           if hits else "No documents are used.")
    prompt = (f"{block}Research topic: {args['research_topic']}.{field_line}\nPropose 3-5 research directions. For each: "
              "title, a testable research question, objectives, feasible methods/datasets/evaluation, feasibility, "
              f"and challenges. {rel}")
    text = ("> **These are suggestions, not verified findings.** Check the literature before committing.\n\n"
            + ctx.llm_chat(build_messages(prompt), temperature=0.7))
    return ToolOutput(text=text, warnings=warnings, grounded=bool(hits))


# ------------------------------------------------------------------ H. analyze_knowledge_gaps
def analyze_knowledge_gaps(args, ctx):
    warnings = []
    hits = _retrieve(ctx, args["topic"], args["selected_document_ids"], max(8, ctx.settings.top_k))
    if not hits:
        raise InsufficientEvidence("The selected documents contain no relevant passages on this topic, so gaps "
                                   "cannot be assessed.")
    block = _evidence(ctx, hits, warnings)
    prompt = (f"{block}\n\nTopic: {args['topic']}\nUsing ONLY these passages, list: (1) questions they leave "
              "unanswered, (2) conflicting or inconsistent statements (cite both labels), (3) areas that may merit "
              "further investigation. Phrase everything relative to these passages.")
    ndocs = len({h.record.document_id for h in hits})
    note = (f"\n\n> *Scope: based only on {len(hits)} retrieved passage(s) from {ndocs} document(s). This is not a "
            "claim about gaps in the wider academic literature.*")
    return ToolOutput(text=ctx.llm_chat(build_messages(prompt)) + note, warnings=warnings, grounded=True)


# ------------------------------------------------------------------ I. create_study_plan
def create_study_plan(args, ctx):
    prompt = (f"Today is {date.today().isoformat()}. Create a realistic study plan.\nSubjects: {', '.join(args['subjects'])}\n"
              f"Available time: {args['available_study_time']}\nDeadline: {args['deadline']}\n"
              f"Goals: {args['learning_goals'] or 'not specified'}\n"
              "Include a weekly/daily schedule, revision sessions, practice, and review checkpoints. Do not exceed the "
              "stated available time; if it is unrealistic for the deadline, say so and suggest priorities.")
    return ToolOutput(text=ctx.llm_chat(build_messages(prompt), temperature=0.4), grounded=False)


# ------------------------------------------------------------------ J. save_research_note
def _note_preview(args):
    return f"Save note '{args['title']}' to project {args['project_id']}:\n\n{args['content'][:600]}"


def save_research_note(args, ctx):
    refs = [r[:200] for r in args["source_references"]]
    nid = ctx.db.add_note(ctx.user_id, args["project_id"], args["title"], args["content"], refs)  # insert-only
    return ToolOutput(data={"note_id": nid, "title": args["title"], "project_id": args["project_id"]},
                      text=f"Saved note '{args['title']}'.")


# ------------------------------------------------------------------ registry
def build_default_registry():
    D = ArgSpec
    reg = ToolRegistry()
    ids = lambda: D("selected_document_ids", "list[str]", "Document IDs to use (empty = whole library)", default=[])
    reg.register(ToolSpec("search_documents", "Search authorised documents and return relevant passages.",
        [D("query", "str", required=True, max_len=1000), ids(), D("top_k", "int", default=5, min=1, max=20)],
        search_documents, doc_args=("selected_document_ids",)))
    reg.register(ToolSpec("summarize_document", "Summarise a document (map-reduce for long documents).",
        [D("document_id", "str", required=True, max_len=64),
         D("summary_length", "enum", choices=("short", "medium", "long"), default="medium"),
         D("focus", "str", default="", max_len=300)], summarize_document, doc_args=("document_id",)))
    reg.register(ToolSpec("explain_concept", "Explain a concept at a chosen level, grounded in documents if available.",
        [D("topic", "str", required=True, max_len=1500),
         D("difficulty_level", "enum", choices=("beginner", "intermediate", "advanced"), default="intermediate"),
         D("optional_document_ids", "list[str]", default=[])], explain_concept, doc_args=("optional_document_ids",)))
    reg.register(ToolSpec("generate_flashcards", "Create Q&A flashcards grounded in evidence.",
        [D("topic", "str", required=True, max_len=500), ids(), D("count", "int", default=10, min=1, max=50)],
        generate_flashcards, doc_args=("selected_document_ids",)))
    reg.register(ToolSpec("generate_quiz", "Create MCQ and short-answer questions with answer key and sources.",
        [D("topic", "str", required=True, max_len=500), ids(),
         D("difficulty", "enum", choices=("easy", "medium", "hard"), default="medium"),
         D("question_count", "int", default=10, min=1, max=30)], generate_quiz, doc_args=("selected_document_ids",)))
    reg.register(ToolSpec("compare_documents", "Compare 2-5 documents using evidence from each.",
        [D("document_ids", "list[str]", required=True, max_items=5),
         D("comparison_topic", "str", default="overall", max_len=500)], compare_documents, doc_args=("document_ids",)))
    reg.register(ToolSpec("generate_research_ideas", "Suggest research directions, questions and methods.",
        [D("research_topic", "str", required=True, max_len=500), ids(), D("research_field", "str", default="", max_len=100)],
        generate_research_ideas, doc_args=("selected_document_ids",)))
    reg.register(ToolSpec("analyze_knowledge_gaps", "Find unanswered questions and conflicts within selected documents.",
        [D("topic", "str", required=True, max_len=500), ids()], analyze_knowledge_gaps, doc_args=("selected_document_ids",)))
    reg.register(ToolSpec("create_study_plan", "Create a realistic study schedule.",
        [D("subjects", "list[str]", required=True), D("available_study_time", "str", required=True, max_len=200),
         D("deadline", "str", required=True, max_len=100), D("learning_goals", "str", default="", max_len=500)],
        create_study_plan))
    reg.register(ToolSpec("save_research_note", "Save a user-approved note to a project (insert-only; asks confirmation).",
        [D("project_id", "str", required=True, max_len=64), D("title", "str", required=True, max_len=120),
         D("content", "str", required=True, max_len=20000), D("source_references", "list[str]", default=[], max_items=50)],
        save_research_note, requires_confirmation=True, preview=_note_preview))
    return reg
