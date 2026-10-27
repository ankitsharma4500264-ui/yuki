# Deploying YUKI: GitHub + Vercel (+ a backend host)

## The shape of the deployment
```
Browser ──► Vercel (React app, static)  ──►  Backend (Docker: FastAPI + ChromaDB + SQLite)  ──►  Groq API
                                              needs a persistent disk                         (chat, STT, TTS)
```
**Vercel hosts the web app. The Python backend runs on a Docker host** (Render blueprint included; Railway, Fly.io or any VPS also work).
Why not run the backend as Vercel functions? Serverless functions have no persistent disk (your PDFs, vectors and chats would vanish), and the multilingual embedding model + PyTorch are far beyond Vercel's function size limit. If you want everything on Vercel, the backend would need hosted Postgres/pgvector and a hosted embeddings API: a real rewrite, so ask if you want it.

## 1. Push to GitHub
```bash
cd yuki-agent
git init && git add . && git commit -m "YUKI agent"
git branch -M main
git remote add origin https://github.com/<you>/yuki.git && git push -u origin main
```
- `.gitignore` already excludes `.env`, `data/` and `node_modules`. A test (`TestRepoHygiene`) fails if an `xai-…` key appears in any committed file.
- **Never paste your Groq key into code, chat or GitHub.** If a key ever leaks, revoke it at https://console.groq.com/keys and create a new one.
- CI (`.github/workflows/ci.yml`) runs the Python tests on 3.11/3.12 and builds the frontend on every push/PR. After your first `npm install` in `frontend/`, commit `package-lock.json` and switch CI to `npm ci`.

## 2. Backend (Render example)
1. Render → **New → Blueprint** → pick your GitHub repo (it reads `render.yaml`).
2. In the dashboard set: `GROQ_API_KEY` (your key), `YUKI_ALLOWED_ORIGINS` (your Vercel URL, e.g. `https://yuki.vercel.app`). `YUKI_SECRET_KEY` is generated for you.
3. Wait for the first build (PyTorch + the embedding model are baked into the image, so it takes a while). Open `https://<service>.onrender.com/api/health`: `status` should be `ok` and `components.model.ok` true.

Notes: the blueprint is untested by me; check its instance size (needs about 1-2 GB RAM) and disk before deploying. The disk keeps documents/chats across deploys. Other hosts: `docker compose up --build` on any VPS, or `docker build` + your platform's volume at `/data`.

## 3. Frontend on Vercel
1. Vercel → **Add New → Project** → import the repo → **Root Directory = `frontend`** (Vite is auto-detected). If you leave the root as `/`, the root `vercel.json` builds the `frontend` folder for you, so either works.
2. **Environment Variable** `VITE_API_BASE` = your backend URL (e.g. `https://yuki-agent-api.onrender.com`, no trailing slash). Deploy.
3. Add your final Vercel URL to the backend's `YUKI_ALLOWED_ORIGINS`. For preview deployments, also set `YUKI_ALLOWED_ORIGIN_REGEX` (e.g. `https://yuki-.*\.vercel\.app`).

*Alternative (same-origin proxy):* copy `frontend/vercel.rewrite.example.json` over `frontend/vercel.json`, put your backend host in it, and leave `VITE_API_BASE` empty. The browser then only talks to Vercel, so there is no CORS setup. I haven't verified how Vercel's proxy treats very large uploads or long streamed answers; if either misbehaves, use the `VITE_API_BASE` direct mode above.

Microphone access needs HTTPS (Vercel provides it; `localhost` also works). `vercel.json` sets `Permissions-Policy: microphone=(self)`.

## 4. Security checklist for a public deploy
- `YUKI_AUTH_MODE=anon` and a strong `YUKI_SECRET_KEY` (the server refuses to start without it). **Never expose `local` mode publicly**: everyone would share one user and see each other's documents.
- Anonymous mode gives each browser its own isolated identity (a signed token in localStorage). It is not a login: clearing browser data loses access to that browser's documents. For real accounts, replace the `user()` dependency in `yuki_agent/api.py` with your auth provider's token check (Clerk, Auth0, Supabase...) and return that user's id.
- Limits (all in memory, per process, so keep one instance): per-user requests (`YUKI_RATE_LIMIT_PER_MIN`), a global chat+voice budget across all users (`YUKI_GLOBAL_RATE_LIMIT_PER_MIN`, protects your xAI credits), a document cap per user (`YUKI_MAX_DOCS_PER_USER`) and upload/audio size caps.
- The Groq key exists only in the backend's environment. The browser never sees it.
- Voice/Chat usage costs money on your xAI account: set a spending limit in the xAI console.

## 5. Costs (from xAI's pricing page when I checked, Oct 2026: verify before relying on it)
Groq bills per token and per audio hour; check https://groq.com/pricing for your model (free tier has rate limits). A typical Q&A with 5 retrieved passages is a few thousand tokens.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Browser console: CORS error | Add the exact Vercel origin to `YUKI_ALLOWED_ORIGINS` (scheme + host, no path/slash) |
| 401 on every request | Backend is in `anon` mode and the token is invalid: clear site data; check `YUKI_SECRET_KEY` didn't change |
| Health says `model.ok: false` | `GROQ_API_KEY` missing/invalid, or `GROQ_MODEL` not available to your key (see `detail`) |
| Vercel: `npm install` / build failed, or exit code 127 | 127 means a command or file was not found. The root `vercel.json` now builds inline (no script file needed), prints the Node/npm versions and the folder it found, retries the install once, and explains what is missing. Open the failed deployment's log and read the lines just above the final error. |
| Many visitors get 429 on first load | Set `YUKI_PROXY_HOPS` (Render = 1; Vercel rewrites + Render = 2) so the per-IP limit sees real visitors, not your proxy |
| Mic button missing | No server voice (no key) and the browser lacks Web Speech (e.g. Firefox); use Chrome/Edge/Safari or set `GROQ_API_KEY` |
| Hindi answers come back in English | Pick "हिन्दी" in the voice bar to force Hindi, or write the question in Devanagari |
| Out-of-memory on the host | Use an instance with more RAM (embeddings model + PyTorch) |


## Alternative: one Vercel project with two services
The repo-root `vercel.json` deploys the FastAPI backend (`yuki-agent`, public only under `/api/*`) and the Vite frontend (`frontend`, everything else) as one project on one domain; the browser calls `/api` on the same origin, so `VITE_API_BASE` stays empty and CORS is not involved. There are no service-to-service calls, so no `bindings` are declared.
- Set in the Vercel project: `GROQ_API_KEY`, `YUKI_SECRET_KEY` (32+ chars). `app.py` defaults the data dir to `/tmp`, auth mode to `anon` and `YUKI_PROXY_HOPS=1` when running on Vercel.
- Local test: `vercel dev` (runs both services).
- Caveats: Vercel's filesystem is ephemeral, so uploaded documents, vectors and chats in SQLite/Chroma are lost when an instance is recycled, and the first request after a cold start downloads the small embedding model (~220 MB) into /tmp, so it is slow. The Vercel build uses the lean `requirements.txt` (fastembed + in-memory vectors); Docker/Render use `requirements-full.txt` (Chroma + PyTorch). For persistent data keep the backend on Render (see above) and use only the frontend on Vercel.
