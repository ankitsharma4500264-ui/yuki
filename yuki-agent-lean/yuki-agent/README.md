# YUKI Agent

Tool-using study & research agent for the YUKI website, in **English and Hindi**, with **voice in and out**.
Model: **Groq** with one `GROQ_API_KEY` (Grok/xAI and local Ollama still work as options). Retrieval is local: multilingual Sentence Transformers + ChromaDB + SQLite.
Windows steps: [SETUP_WINDOWS.md](SETUP_WINDOWS.md) · Deploy to GitHub + Vercel: [DEPLOY.md](DEPLOY.md).

## What works where
| Feature | Needs |
|---|---|
| Chat, RAG Q&A with citations, summaries, explain, flashcards, quiz, compare, research ideas, knowledge gaps, study plans, notes, sessions, history, privacy controls | **Local only** (Ollama + `pip` packages) |
| Real PDF extraction | `pymupdf` |
| Groq model + Whisper speech-to-text (Hindi/English) | **your** `GROQ_API_KEY` in `.env` (never in git or the browser) |
| Browser voice fallback (no key) | Web Speech API in Chrome/Edge/Safari (Chrome's recogniser sends audio to Google) |
| Other cloud model (optional) | any OpenAI-compatible endpoint (`YUKI_PROVIDER=openai_compat`) |
| Not included | OCR for scanned PDFs, cross-encoder reranking (a light lexical rerank is built in), token-by-token streaming |

## Groq, Hindi and voice
- **Groq:** `YUKI_PROVIDER=auto` uses Groq when `GROQ_API_KEY` is set (model `GROQ_MODEL`, default `llama-3.3-70b-versatile`, over Groq's OpenAI-compatible endpoint), then Grok if only `XAI_API_KEY` is set, else Ollama. `/api/health` checks the key and that the model is available to it.
- **Grok (optional):** `YUKI_PROVIDER=auto` uses Grok when `XAI_API_KEY` is set (model `XAI_MODEL`, default `grok-4.7`, over xAI's OpenAI-compatible endpoint), else Ollama. `/api/health` checks the key and that the model is available to it.
- **Hindi:** Devanagari questions are detected and answered in Hindi (or force it with the language selector); intents work in Hindi and common Hinglish ("सारांश दो", "quiz banao"); fixed messages are localised; retrieval uses a multilingual embedding model, so Hindi questions find English passages and vice versa. Romanised Hinglish is treated as English-mode but the model mirrors your style.
- **Voice:** mic button records, the backend transcribes with Groq Whisper (`GROQ_STT_MODEL`, Hindi/English auto-detected or forced). Groq has no text-to-speech for Hindi, so replies are read aloud with the browser's voice, or with Grok TTS (`naksh`) if `XAI_API_KEY` is also set (change with `YUKI_VOICE_EN/HI`). Spoken replies are shortened and stripped of markdown and `[S#]` markers. No key → the browser's Web Speech API is used instead.
- Endpoints: `GET /api/voice/config`, `POST /api/voice/transcribe` (multipart `language`, then `file`), `POST /api/voice/speak` (`{text, language}` → audio).
- Audio is not stored by this app. Voice is billed by Groq/xAI per use.

## Architecture (`yuki_agent/`)
| Component | File |
|---|---|
| Agent Controller | `controller.py` |
| Intent Classifier | `intent.py` (rules; optional model routing with JSON tool-call protocol) |
| Tool Registry / Executor | `tools.py` (schemas, validation, ownership check, dedupe, call cap, timeout, confirmation gate) |
| The 10 tools | `tool_impls.py` |
| RAG Retriever / ingestion / stores | `rag.py` |
| Context Builder | `context.py` |
| LLM Provider abstraction | `providers.py` (Ollama, OpenAI-compatible) |
| Citation Manager | `citations.py` |
| Conversation Manager / DB | `conversation.py`, `db.py` |
| Response Formatter + activity log | `formatter.py` |
| Safety & Validation | `safety.py`, `auth.py` (anonymous signed tokens) |
| Hindi/English helpers | `lang.py` |
| Voice (Grok STT/TTS) | `voice.py` |
| FastAPI | `api.py`, wiring in `services.py` |

Flow: validate → resolve session + authorised documents → classify intent → build/validate tool args → execute tool(s) → check results → answer from evidence → strip invalid citations → persist → structured response (answer, sources, tool activity, warnings, status).

## Honesty guarantees (enforced in code, covered by tests)
- Citation labels `[S#]` come only from chunks actually retrieved in that request; unknown markers are removed and reported. Quoted text not found in the evidence triggers a warning.
- No relevant evidence → "cannot be established from the selected sources" and **no model call** (so nothing can be invented).
- Tool activity lists only steps that really ran, with real durations. No hidden reasoning is exposed.
- A failed tool/model/vector DB returns `status: "error"`; saving a note reports success only after SQLite confirms.
- Document text is untrusted: wrapped in `<evidence>`, delimiter/label spoofing neutralised, injection-like text flagged, system rules say never obey it. The agent has **no** delete/shell/code tools; notes are insert-only and require user confirmation (`POST /api/agent/confirm`).

## API
`POST /api/agent/chat` · `POST /api/agent/chat/stream` (SSE: `activity` then `final`) · `POST /api/agent/confirm` · `GET /api/agent/sessions[/{id}]` · `DELETE /api/agent/sessions[/{id}]` · `DELETE /api/agent/user-data?confirm=true` · `GET /api/agent/tools` · `GET /api/health` · `POST|GET|DELETE /api/documents` · `POST|GET /api/projects` · `GET /api/projects/{id}/notes`
Chat body: `message`, `session_id`, `selected_document_ids`, `project_id`, `preferences {language,length,tone}`, `regenerate`.
Response: `answer`, `sources[]`, `session_id`, `tool_activity[]`, `status`, `warnings[]`, `structured` (flashcards/quiz data), `pending_action`, `error`, `metadata`.
`status` ∈ completed | needs_input | needs_confirmation | insufficient_evidence | error | cancelled.

## Integrating with your existing app
- **Documents:** if you already have upload/RAG endpoints, skip `/api/documents` and call `RagService.ingest_file(user_id, filename, path)` from yours; the agent only needs documents indexed through that service.
- **Auth:** `YUKI_AUTH_MODE=local` is single-user and for localhost only; `anon` gives each browser an isolated signed identity (use it for any public deploy; the server needs `YUKI_SECRET_KEY`).
- **Real accounts:** replace the `user()` dependency in `api.py` with your session/JWT check and return your user id; every query is already scoped by it. With `YUKI_API_TOKEN` set, a trusted gateway sends `Authorization: Bearer <token>` + `X-User-Id`.
- **Streaming:** the server streams *activity events* live; the final answer arrives as one event (the model call is not token-streamed). Cancel = abort the fetch; work stops at the next checkpoint.
- Never put API keys in React code; the browser only talks to this API.

## Tuning RAG
Defaults: 700-token chunks (approximated as 0.75 words/token), 100 overlap, top-5, min score 0.25. Chunks never span pages, so page numbers are exact. Evaluate on your real PDFs by lowering/raising `YUKI_MIN_SCORE` and `YUKI_TOP_K`; check the `score` values returned in `sources`.

## Tests
- Backend: `python -m pytest -q` (fake model + in-memory store; real-Ollama test is opt-in with `YUKI_INTEGRATION=1`).
- Frontend: `cd frontend && npm test` (fuzzes the markdown renderer: no hangs, no raw HTML). CI runs both.

## Known limits
- Tool timeouts stop *waiting*; a model call already in flight finishes in the background.
- Small local models may return malformed JSON for quizzes/flashcards: the tool retries once, validates every item, drops invalid or duplicate ones and warns when fewer than requested are produced.
- Tests use a fake model and an in-memory vector store; the real Ollama test is opt-in (`YUKI_INTEGRATION=1`).
