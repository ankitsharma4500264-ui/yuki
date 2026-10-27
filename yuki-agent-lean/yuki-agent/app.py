"""Vercel entrypoint (service "yuki-agent"): exposes the FastAPI app as `app`.
Vercel serves it at /api/* (see the root vercel.json), and the routes in yuki_agent/api.py already start with /api.
On Vercel only /tmp is writable (and it is ephemeral), so data and the model cache are pointed there."""
import os

if os.getenv("VERCEL"):
    os.environ.setdefault("YUKI_DATA_DIR", "/tmp/yuki-data")
    os.environ.setdefault("HF_HOME", "/tmp/hf")
    os.environ.setdefault("YUKI_AUTH_MODE", "anon")      # public deploy: each browser gets its own signed identity
    os.environ.setdefault("YUKI_PROXY_HOPS", "1")        # real visitor IP comes from X-Forwarded-For
    # Keep the bundle small (500 MB limit): ONNX embeddings instead of PyTorch, in-memory vectors instead of Chroma.
    os.environ.setdefault("YUKI_EMBEDDING_BACKEND", "fastembed")
    os.environ.setdefault("YUKI_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  # Hindi + English
    os.environ.setdefault("YUKI_VECTOR_BACKEND", "memory")
    os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp/fastembed")

from yuki_agent.api import create_app  # noqa: E402  (env must be set before the app reads its settings)

app = create_app()
