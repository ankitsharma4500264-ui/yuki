"""Model-provider abstraction. Switch providers with YUKI_PROVIDER; the agent never changes."""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

from .errors import ProviderError, ProviderUnavailable


def http_request(url, data=None, headers=None, timeout=60.0, method="POST"):
    """Returns (body_bytes, response_headers). Error messages never include the request, headers or API keys."""
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), r.headers
    except urllib.error.HTTPError as e:
        msg = {401: "The API key was rejected (HTTP 401). Check your API key setting (GROQ_API_KEY for Groq, XAI_API_KEY for Grok).",
               403: "The API key is not allowed to use this endpoint or model (HTTP 403).",
               404: "The model or endpoint was not found (HTTP 404). Check the model name.",
               413: "The upload is too large for the service (HTTP 413).",
               429: "The service is rate-limiting requests (HTTP 429). Try again shortly."}.get(
            e.code, f"The model server returned HTTP {e.code}.")
        raise ProviderError(msg, status=e.code)
    except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError, OSError):
        raise ProviderUnavailable("Could not reach the model server (it may be offline or timed out).")


def _http(url, payload=None, headers=None, timeout=60.0, method="POST"):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    body, _ = http_request(url, data, {"Content-Type": "application/json", **(headers or {})}, timeout, method)
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError:
        raise ProviderError("The model server returned an unreadable response.")


class LLMProvider:
    name = "base"
    model = ""

    def chat(self, messages, json_mode=False, temperature=0.2, timeout=60.0):  # pragma: no cover
        raise NotImplementedError

    def health(self):  # pragma: no cover
        return {"ok": False, "provider": self.name, "model": self.model}


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, host, model):
        self.host, self.model = host.rstrip("/"), model

    def chat(self, messages, json_mode=False, temperature=0.2, timeout=60.0):
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": temperature}}
        if json_mode:
            payload["format"] = "json"
        try:
            d = _http(f"{self.host}/api/chat", payload, timeout=timeout)
        except ProviderError as e:
            if e.status == 404:
                raise ProviderError(f"Model '{self.model}' is not installed. Run: ollama pull {self.model}",
                                    status=404)
            raise
        try:
            return d["message"]["content"]
        except (KeyError, TypeError):
            raise ProviderError("Unexpected response format from Ollama.")

    def health(self):
        try:
            d = _http(f"{self.host}/api/tags", timeout=3.0, method="GET")
        except ProviderError as e:
            return {"ok": False, "provider": self.name, "model": self.model, "detail": str(e)}
        names = {m.get("name", "") for m in d.get("models", [])}
        have = self.model in names or f"{self.model}:latest" in names
        return {"ok": have, "provider": self.name, "model": self.model, "model_available": have,
                "detail": None if have else f"Run: ollama pull {self.model}"}


class OpenAICompatProvider(LLMProvider):
    """Any OpenAI-compatible /chat/completions server (optional, needs a key you supply)."""
    name = "openai_compat"

    def __init__(self, base_url, api_key, model):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model

    def chat(self, messages, json_mode=False, temperature=0.2, timeout=60.0):
        if not (self.api_key and self.model):
            raise ProviderUnavailable("Cloud provider is not configured (YUKI_CLOUD_API_KEY / YUKI_CLOUD_MODEL).")
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        d = _http(f"{self.base_url}/chat/completions", payload,
                  headers={"Authorization": f"Bearer {self.api_key}"}, timeout=timeout)
        try:
            return d["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("Unexpected response format from the cloud provider.")

    def health(self):
        ok = bool(self.api_key and self.model)
        return {"ok": ok, "provider": self.name, "model": self.model,
                "detail": None if ok else "API key or model not set"}


class XAIProvider(LLMProvider):
    """Grok via xAI's OpenAI-compatible Chat Completions endpoint (https://docs.x.ai)."""
    name = "xai"

    def __init__(self, api_key, model="grok-4.7", base_url="https://api.x.ai/v1", reasoning_effort=""):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort

    def _auth(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def chat(self, messages, json_mode=False, temperature=0.2, timeout=60.0):
        if not self.api_key:
            raise ProviderUnavailable("Grok is not configured: set XAI_API_KEY.")
        full = {"model": self.model, "messages": messages, "temperature": temperature}
        if self.reasoning_effort:
            full["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            full["response_format"] = {"type": "json_object"}
        minimal = {"model": self.model, "messages": messages}   # fallback if a parameter is rejected
        try:
            d = _http(f"{self.base_url}/chat/completions", full, self._auth(), timeout)
        except ProviderError as e:
            if e.status != 400 or isinstance(e, ProviderUnavailable):
                raise
            d = _http(f"{self.base_url}/chat/completions", minimal, self._auth(), timeout)
        try:
            return d["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ProviderError("Unexpected response format from Grok.")

    def health(self):
        if not self.api_key:
            return {"ok": False, "provider": self.name, "model": self.model, "detail": "XAI_API_KEY is not set"}
        try:
            d = _http(f"{self.base_url}/models", headers=self._auth(), timeout=5.0, method="GET")
        except ProviderError as e:
            return {"ok": False, "provider": self.name, "model": self.model, "detail": str(e)}
        ids = {m.get("id", "") for m in d.get("data", []) if isinstance(m, dict)}
        have = (not ids) or self.model in ids or any(i.startswith(self.model) for i in ids)
        return {"ok": have, "provider": self.name, "model": self.model, "model_available": have,
                "detail": None if have else f"Model '{self.model}' is not available to this key"}


class GroqProvider(LLMProvider):
    """Groq's OpenAI-compatible Chat Completions endpoint (https://console.groq.com/docs)."""
    name = "groq"

    def __init__(self, api_key, model="llama-3.3-70b-versatile", base_url="https://api.groq.com/openai/v1"):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")

    def _auth(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def chat(self, messages, json_mode=False, temperature=0.2, timeout=60.0):
        if not self.api_key:
            raise ProviderUnavailable("Groq is not configured: set GROQ_API_KEY.")
        full = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_mode:
            full["response_format"] = {"type": "json_object"}
        minimal = {"model": self.model, "messages": messages}   # fallback if a parameter is rejected
        try:
            d = _http(f"{self.base_url}/chat/completions", full, self._auth(), timeout)
        except ProviderError as e:
            if e.status != 400 or isinstance(e, ProviderUnavailable):
                raise
            d = _http(f"{self.base_url}/chat/completions", minimal, self._auth(), timeout)
        try:
            return d["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ProviderError("Unexpected response format from Groq.")

    def health(self):
        if not self.api_key:
            return {"ok": False, "provider": self.name, "model": self.model, "detail": "GROQ_API_KEY is not set"}
        try:
            d = _http(f"{self.base_url}/models", headers=self._auth(), timeout=5.0, method="GET")
        except ProviderError as e:
            return {"ok": False, "provider": self.name, "model": self.model, "detail": str(e)}
        ids = {m.get("id", "") for m in d.get("data", []) if isinstance(m, dict)}
        have = (not ids) or self.model in ids
        return {"ok": have, "provider": self.name, "model": self.model, "model_available": have,
                "detail": None if have else f"Model '{self.model}' is not available to this key"}


def build_provider(settings):
    p = settings.provider
    if p == "auto":   # Groq first, then Grok, then local Ollama
        p = "groq" if settings.groq_api_key else "xai" if settings.xai_api_key else "ollama"
    if p == "groq":
        return GroqProvider(settings.groq_api_key, settings.groq_model, settings.groq_base_url)
    if p == "xai":
        return XAIProvider(settings.xai_api_key, settings.xai_model, settings.xai_base_url, settings.xai_reasoning_effort)
    if p == "ollama":
        return OllamaProvider(settings.ollama_host, settings.ollama_model)
    if p == "openai_compat":
        return OpenAICompatProvider(settings.cloud_base_url, settings.cloud_api_key, settings.cloud_model)
    raise ValueError(f"Unknown YUKI_PROVIDER '{settings.provider}'")
