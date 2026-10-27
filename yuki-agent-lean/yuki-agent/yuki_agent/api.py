"""FastAPI routes.  Run:  uvicorn yuki_agent.api:create_app --factory --port 8000"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import queue
import re
import threading
import uuid
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .auth import new_anonymous_user, sign_token, verify_token
from .controller import ChatRequest
from .errors import PermissionDenied, ProviderError, ProviderUnavailable, ValidationError, YukiError
from .safety import RateLimiter, safe_filename, validate_upload
from .services import Services, build_services
from .voice import VoiceService

USER_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")


# ------------------------------------------------------------------ schemas
class ChatBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=20000)
    session_id: Optional[str] = None
    selected_document_ids: Optional[List[str]] = Field(default=None, max_length=20)  # null = keep the session's selection
    project_id: Optional[str] = None
    preferences: Optional[dict] = None
    regenerate: bool = False


class ConfirmBody(BaseModel):
    pending_id: str
    approve: bool


class SpeakBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    language: Optional[str] = None


class ProjectBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


def _to_chat_request(b: ChatBody):
    return ChatRequest(b.message, b.session_id, b.selected_document_ids, b.project_id, b.preferences, b.regenerate)


def create_app(services: Optional[Services] = None) -> FastAPI:
    svc = services or build_services()
    s = svc.settings
    voice = svc.voice or VoiceService(s)
    if s.auth_mode == "anon" and not s.api_token and len(s.secret_key) < 32:
        raise RuntimeError("YUKI_AUTH_MODE=anon needs YUKI_SECRET_KEY of at least 32 characters "
                           "(generate one: python -c \"import secrets; print(secrets.token_urlsafe(48))\").")
    limiter = RateLimiter(s.rate_limit_per_min)
    auth_limiter = RateLimiter(20)
    global_limiter = RateLimiter(s.global_rate_limit_per_min)
    app = FastAPI(title="YUKI Agent API", version="1.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in s.allowed_origins.split(",") if o.strip()],
                       allow_origin_regex=s.allowed_origin_regex or None, allow_methods=["*"], allow_headers=["*"])

    # -- auth modes (see README "Deploy"):
    #    gateway: YUKI_API_TOKEN set -> a trusted backend sends Bearer <token> (+ X-User-Id).
    #    anon:    YUKI_AUTH_MODE=anon -> each browser gets a signed anonymous token (POST /api/auth/anonymous).
    #    local:   single user "local"; ONLY for localhost development, never expose it publicly.
    def user(request: Request) -> str:
        if not s.api_token and s.auth_mode == "anon":
            auth = request.headers.get("authorization", "")
            uid = verify_token(s.secret_key, auth[7:]) if auth.startswith("Bearer ") else None
            if not uid:
                raise HTTPException(401, "Missing or invalid session token")
            if not limiter.allow(uid):
                raise HTTPException(429, "Too many requests. Please slow down.")
            return uid
        if s.api_token:
            if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {s.api_token}"):
                raise HTTPException(401, "Unauthorized")
            uid = request.headers.get("x-user-id", "local")
            if not USER_RE.match(uid):
                raise HTTPException(400, "Invalid user id")
        else:
            uid = "local"
        if not limiter.allow(uid):
            raise HTTPException(429, "Too many requests. Please slow down.")
        return uid

    def guard(fn, *a):
        try:
            return fn(*a)
        except PermissionDenied as e:
            raise HTTPException(404, str(e))
        except ValidationError as e:
            raise HTTPException(400, str(e))
        except YukiError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/auth/anonymous")
    def anonymous(request: Request):
        if s.api_token or s.auth_mode != "anon":
            raise HTTPException(404, "Not found")
        if not auth_limiter.allow(client_ip(request)):
            raise HTTPException(429, "Too many requests.")
        uid = new_anonymous_user()
        return {"token": sign_token(s.secret_key, uid, s.anon_token_days), "user_id": uid}

    def client_ip(request: Request) -> str:
        """Socket IP, or the address the Nth trusted proxy from the right saw (the left-most entries can be forged)."""
        if s.proxy_hops > 0:
            parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
            if len(parts) >= s.proxy_hops:
                return parts[-s.proxy_hops][:64]
        return request.client.host if request.client else "?"

    def costly(uid: str = Depends(user)) -> str:
        """Endpoints that spend xAI credits also share one global budget, so minting many anonymous users can't drain it."""
        if not global_limiter.allow("*"):
            raise HTTPException(429, "The service is busy right now. Please try again in a minute.")
        return uid

    def provider_guard(fn, *a):
        try:
            return fn(*a)
        except ValidationError as e:
            raise HTTPException(400, str(e))
        except ProviderUnavailable as e:
            raise HTTPException(503, str(e))
        except ProviderError as e:
            raise HTTPException(502, str(e))

    # ------------------------------------------------------------- voice (Hindi + English)
    @app.get("/api/voice/config")
    def voice_config(_: str = Depends(user)):
        return {"server_stt": voice.stt_enabled, "server_tts": voice.tts_enabled, "languages": ["en", "hi"]}

    @app.post("/api/voice/transcribe")
    async def transcribe(file: UploadFile = File(...), language: str = Form(""), uid: str = Depends(costly)):
        data = await file.read(s.max_audio_mb * 1024 * 1024 + 1)
        if len(data) > s.max_audio_mb * 1024 * 1024:
            raise HTTPException(413, f"Audio exceeds the {s.max_audio_mb} MB limit.")
        return await asyncio.to_thread(provider_guard, voice.transcribe, data, safe_filename(file.filename), language or None)

    @app.post("/api/voice/speak")
    async def speak(body: SpeakBody, uid: str = Depends(costly)):
        audio, ctype = await asyncio.to_thread(provider_guard, voice.synthesize, body.text, body.language)
        return Response(content=audio, media_type=ctype, headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------- health / tools
    @app.get("/api/health")
    def health():
        out = {"status": "ok", "components": {}}
        try:
            svc.db.ping()
            out["components"]["database"] = {"ok": True}
        except Exception:
            out["components"]["database"] = {"ok": False}
        try:
            svc.store.count()
            out["components"]["vector_store"] = {"ok": True}
        except Exception:
            out["components"]["vector_store"] = {"ok": False}
        out["components"]["model"] = svc.llm.health()
        if not all(c.get("ok") for c in out["components"].values()):
            out["status"] = "degraded"
        return out

    @app.get("/api/agent/tools")
    def tools(_: str = Depends(user)):
        return {"tools": svc.controller.registry.describe()}

    # ------------------------------------------------------------- chat
    @app.post("/api/agent/chat")
    async def chat(body: ChatBody, uid: str = Depends(costly)):
        resp = await asyncio.to_thread(svc.controller.handle, _to_chat_request(body), uid)
        return resp.to_dict()

    @app.post("/api/agent/chat/stream")
    async def chat_stream(body: ChatBody, uid: str = Depends(costly)):
        """Server-sent events: `activity` events as steps really happen, then one `final` event."""
        q: "queue.Queue" = queue.Queue()
        cancel = threading.Event()

        def worker():
            try:
                r = svc.controller.handle(_to_chat_request(body), uid, on_event=lambda e: q.put(("activity", e)),
                                          cancel=cancel)
                q.put(("final", r.to_dict()))
            except Exception:
                q.put(("error", {"error": "Something went wrong."}))
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()

        async def gen():
            try:
                while True:
                    item = await asyncio.to_thread(q.get)
                    if item is None:
                        break
                    yield f"event: {item[0]}\ndata: {json.dumps(item[1])}\n\n"
            finally:
                cancel.set()    # client disconnected / aborted -> stop work at the next checkpoint

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/agent/confirm")
    async def confirm(body: ConfirmBody, uid: str = Depends(user)):
        resp = await asyncio.to_thread(guard, svc.controller.resolve_pending, uid, body.pending_id, body.approve)
        return resp.to_dict()

    # ------------------------------------------------------------- sessions
    @app.get("/api/agent/sessions")
    def sessions(uid: str = Depends(user)):
        return {"sessions": svc.db.list_sessions(uid)}

    @app.get("/api/agent/sessions/{session_id}")
    def session(session_id: str, uid: str = Depends(user)):
        sess = svc.db.get_session(uid, session_id)
        if not sess:
            raise HTTPException(404, "Session not found")
        return {"session": sess, "messages": svc.db.get_messages(uid, session_id)}

    @app.delete("/api/agent/sessions/{session_id}")
    def delete_session(session_id: str, uid: str = Depends(user)):
        if not svc.db.get_session(uid, session_id):
            raise HTTPException(404, "Session not found")
        svc.db.delete_session(uid, session_id)
        return {"deleted": True}

    @app.delete("/api/agent/sessions")
    def clear_history(uid: str = Depends(user)):
        svc.db.delete_all_sessions(uid)
        return {"deleted": True}

    @app.delete("/api/agent/user-data")
    def delete_user_data(confirm: bool = False, uid: str = Depends(user)):
        if not confirm:
            raise HTTPException(400, "Pass ?confirm=true to permanently delete all your documents, chats and notes.")
        n = svc.rag.delete_user_data(uid)
        return {"deleted": True, "documents_removed": n}

    # ------------------------------------------------------------- documents (reuse yours if you already have these)
    @app.post("/api/documents")
    async def upload(file: UploadFile = File(...), uid: str = Depends(user)):
        if len(svc.db.list_documents(uid)) >= s.max_docs_per_user:
            raise HTTPException(400, f"Document limit reached ({s.max_docs_per_user}). Delete some documents first.")
        data = await file.read(s.max_upload_mb * 1024 * 1024 + 1)
        name = safe_filename(file.filename)
        ext = guard(validate_upload, name, len(data), data[:1024], s.max_upload_mb)
        folder = os.path.join(s.upload_dir, re.sub(r"[^A-Za-z0-9_.-]", "_", uid))
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, uuid.uuid4().hex + ext)
        with open(path, "wb") as f:
            f.write(data)
        try:
            doc = await asyncio.to_thread(svc.rag.ingest_file, uid, name, path)
        except YukiError as e:
            os.remove(path)
            raise HTTPException(422, str(e))
        except Exception:
            os.remove(path)
            raise HTTPException(500, "Could not process the document.")
        return {k: doc[k] for k in ("id", "filename", "pages", "chunks", "status", "created_at")}

    @app.get("/api/documents")
    def documents(uid: str = Depends(user)):
        return {"documents": [{k: d[k] for k in ("id", "filename", "pages", "chunks", "status", "created_at")}
                              for d in svc.db.list_documents(uid)]}

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: str, uid: str = Depends(user)):   # user-initiated only; the agent has no delete tool
        guard(svc.rag.delete_document, uid, doc_id)
        return {"deleted": True}

    # ------------------------------------------------------------- projects & notes
    @app.post("/api/projects")
    def create_project(body: ProjectBody, uid: str = Depends(user)):
        return svc.db.create_project(uid, body.name)

    @app.get("/api/projects")
    def projects(uid: str = Depends(user)):
        return {"projects": svc.db.list_projects(uid)}

    @app.get("/api/projects/{project_id}/notes")
    def notes(project_id: str, uid: str = Depends(user)):
        if not svc.db.get_project(uid, project_id):
            raise HTTPException(404, "Project not found")
        return {"notes": svc.db.list_notes(uid, project_id)}

    return app
