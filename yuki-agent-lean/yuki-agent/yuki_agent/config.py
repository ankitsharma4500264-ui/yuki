"""All configuration comes from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _s(name, default):
    v = os.getenv(name)
    return default if v is None or v == "" else v


def _i(name, default):
    try:
        return int(_s(name, default))
    except ValueError:
        return default


def _f(name, default):
    try:
        return float(_s(name, default))
    except ValueError:
        return default


def _b(name, default):
    return str(_s(name, default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # model provider
    provider: str = "ollama"                      # auto | groq | xai | ollama | openai_compat (from_env default: auto)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    cloud_base_url: str = "https://api.openai.com/v1"
    cloud_api_key: str = ""
    cloud_model: str = ""
    groq_api_key: str = ""                         # Groq: chat (Llama/GPT-OSS) + Whisper speech-to-text
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_stt_model: str = "whisper-large-v3-turbo"
    xai_api_key: str = ""                          # Grok (xAI): chat + speech-to-text + text-to-speech
    xai_model: str = "grok-4.7"
    xai_base_url: str = "https://api.x.ai/v1"
    xai_reasoning_effort: str = ""                 # optional; sent only if set
    voice_en: str = "naksh"
    voice_hi: str = "naksh"
    tts_max_chars: int = 1500
    max_audio_mb: int = 10
    # retrieval
    embedding_backend: str = "sentence-transformers"   # sentence-transformers | fastembed (light, no PyTorch) | hash (tests only)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_backend: str = "chroma"                      # chroma | memory (tests only)
    data_dir: str = "./data"
    chunk_tokens: int = 700
    chunk_overlap: int = 100
    top_k: int = 5
    min_score: float = 0.25
    rerank: bool = True
    # agent limits
    max_tool_calls: int = 4
    request_timeout_s: float = 180.0
    tool_timeout_s: float = 150.0
    llm_timeout_s: float = 120.0
    llm_routing: bool = False       # let the model pick a tool when rules are unsure
    history_messages: int = 10
    summary_section_chars: int = 6000
    max_summary_sections: int = 40
    # security
    max_message_chars: int = 4000
    max_upload_mb: int = 25
    rate_limit_per_min: int = 30
    global_rate_limit_per_min: int = 120      # all users together, on chat + voice endpoints (cost guard); 0 = off
    max_docs_per_user: int = 30
    max_pages: int = 1500                     # PDF page cap (bounds extraction/embedding time)
    max_chunks_per_doc: int = 4000
    proxy_hops: int = 0                       # trusted reverse proxies in front (Render=1, Vercel rewrite + Render=2); 0 = use socket IP
    api_token: str = ""
    allowed_origins: str = "http://localhost:5173,http://localhost:3000"
    allowed_origin_regex: str = ""                 # e.g. https://.*\\.vercel\\.app for preview deployments
    auth_mode: str = "local"                       # local (single user, localhost only) | anon (signed anonymous users)
    secret_key: str = ""                           # >= 32 chars; signs anonymous-user tokens
    anon_token_days: int = 180

    @property
    def db_path(self):
        return os.path.join(self.data_dir, "yuki.sqlite3")

    @property
    def chroma_dir(self):
        return os.path.join(self.data_dir, "chroma")

    @property
    def upload_dir(self):
        return os.path.join(self.data_dir, "uploads")

    @classmethod
    def from_env(cls):
        d = cls()
        return cls(
            provider=_s("YUKI_PROVIDER", "auto"),
            ollama_host=_s("OLLAMA_HOST", d.ollama_host).rstrip("/"),
            ollama_model=_s("OLLAMA_MODEL", d.ollama_model),
            cloud_base_url=_s("YUKI_CLOUD_BASE_URL", d.cloud_base_url).rstrip("/"),
            cloud_api_key=_s("YUKI_CLOUD_API_KEY", d.cloud_api_key),
            cloud_model=_s("YUKI_CLOUD_MODEL", d.cloud_model),
            groq_api_key=_s("GROQ_API_KEY", d.groq_api_key),
            groq_model=_s("GROQ_MODEL", d.groq_model),
            groq_base_url=_s("GROQ_BASE_URL", d.groq_base_url).rstrip("/"),
            groq_stt_model=_s("GROQ_STT_MODEL", d.groq_stt_model),
            xai_api_key=_s("XAI_API_KEY", d.xai_api_key),
            xai_model=_s("XAI_MODEL", d.xai_model),
            xai_base_url=_s("XAI_BASE_URL", d.xai_base_url).rstrip("/"),
            xai_reasoning_effort=_s("YUKI_REASONING_EFFORT", d.xai_reasoning_effort),
            voice_en=_s("YUKI_VOICE_EN", d.voice_en),
            voice_hi=_s("YUKI_VOICE_HI", d.voice_hi),
            tts_max_chars=_i("YUKI_TTS_MAX_CHARS", d.tts_max_chars),
            max_audio_mb=_i("YUKI_MAX_AUDIO_MB", d.max_audio_mb),
            embedding_backend=_s("YUKI_EMBEDDING_BACKEND", d.embedding_backend),
            embedding_model=_s("YUKI_EMBEDDING_MODEL", d.embedding_model),
            vector_backend=_s("YUKI_VECTOR_BACKEND", d.vector_backend),
            data_dir=_s("YUKI_DATA_DIR", d.data_dir),
            chunk_tokens=_i("YUKI_CHUNK_TOKENS", d.chunk_tokens),
            chunk_overlap=_i("YUKI_CHUNK_OVERLAP", d.chunk_overlap),
            top_k=_i("YUKI_TOP_K", d.top_k),
            min_score=_f("YUKI_MIN_SCORE", d.min_score),
            rerank=_b("YUKI_RERANK", d.rerank),
            max_tool_calls=_i("YUKI_MAX_TOOL_CALLS", d.max_tool_calls),
            request_timeout_s=_f("YUKI_REQUEST_TIMEOUT_S", d.request_timeout_s),
            tool_timeout_s=_f("YUKI_TOOL_TIMEOUT_S", d.tool_timeout_s),
            llm_timeout_s=_f("YUKI_LLM_TIMEOUT_S", d.llm_timeout_s),
            llm_routing=_b("YUKI_LLM_ROUTING", d.llm_routing),
            history_messages=_i("YUKI_HISTORY_MESSAGES", d.history_messages),
            summary_section_chars=_i("YUKI_SUMMARY_SECTION_CHARS", d.summary_section_chars),
            max_summary_sections=_i("YUKI_MAX_SUMMARY_SECTIONS", d.max_summary_sections),
            max_message_chars=_i("YUKI_MAX_MESSAGE_CHARS", d.max_message_chars),
            max_upload_mb=_i("YUKI_MAX_UPLOAD_MB", d.max_upload_mb),
            rate_limit_per_min=_i("YUKI_RATE_LIMIT_PER_MIN", d.rate_limit_per_min),
            global_rate_limit_per_min=_i("YUKI_GLOBAL_RATE_LIMIT_PER_MIN", d.global_rate_limit_per_min),
            max_docs_per_user=_i("YUKI_MAX_DOCS_PER_USER", d.max_docs_per_user),
            max_pages=_i("YUKI_MAX_PAGES", d.max_pages),
            max_chunks_per_doc=_i("YUKI_MAX_CHUNKS_PER_DOC", d.max_chunks_per_doc),
            proxy_hops=_i("YUKI_PROXY_HOPS", d.proxy_hops),
            api_token=_s("YUKI_API_TOKEN", d.api_token),
            allowed_origins=_s("YUKI_ALLOWED_ORIGINS", d.allowed_origins),
            allowed_origin_regex=_s("YUKI_ALLOWED_ORIGIN_REGEX", d.allowed_origin_regex),
            auth_mode=_s("YUKI_AUTH_MODE", d.auth_mode).lower(),
            secret_key=_s("YUKI_SECRET_KEY", d.secret_key),
            anon_token_days=_i("YUKI_ANON_TOKEN_DAYS", d.anon_token_days),
        )
