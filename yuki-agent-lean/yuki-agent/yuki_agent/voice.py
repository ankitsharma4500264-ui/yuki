"""Server-side Hindi/English voice. Speech-to-text: Groq Whisper (preferred) or xAI. Text-to-speech: xAI only
(Groq has no Hindi TTS, so with Groq alone replies are read by the browser's built-in voice).
The API key never leaves the server. Browsers without a key fall back to the Web Speech API on the client."""
from __future__ import annotations

import json
import uuid

from .errors import ProviderError, ProviderUnavailable, ValidationError
from .lang import LANGS, detect_language, speakable
from .providers import http_request
from .safety import safe_filename

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".opus", ".flac", ".aac", ".mp4", ".m4a", ".mkv"}


def encode_multipart(fields, file_field, filename, content, content_type):
    """Multipart body with the file as the LAST part (xAI ignores fields placed after the file)."""
    boundary = uuid.uuid4().hex
    out = []
    for k, v in fields:
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    out.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{safe_filename(filename)}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n").encode())
    out += [content, f"\r\n--{boundary}--\r\n".encode()]
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


class VoiceService:
    STT_MODEL = "grok-voice-transcribe-2.0"

    def __init__(self, settings):
        self.s = settings

    @property
    def groq_stt(self):   # Groq Whisper is used whenever a Groq key is set (unless xAI is forced)
        return bool(self.s.groq_api_key) and self.s.provider != "xai"

    @property
    def stt_enabled(self):
        return self.groq_stt or bool(self.s.xai_api_key)

    @property
    def tts_enabled(self):
        return bool(self.s.xai_api_key)

    @property
    def enabled(self):
        return self.stt_enabled or self.tts_enabled

    def _auth(self, groq=False):
        key = self.s.groq_api_key if groq else self.s.xai_api_key
        if not key:
            raise ProviderUnavailable("Server voice is not configured (GROQ_API_KEY / XAI_API_KEY).")
        return {"Authorization": f"Bearer {key}"}

    def transcribe(self, audio, filename, language=None):
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in AUDIO_EXTS:
            raise ValidationError("Unsupported audio format. Use WAV, MP3, OGG, FLAC, M4A or MP4.")
        if not audio:
            raise ValidationError("Audio is empty.")
        if language and language not in LANGS:
            raise ValidationError("Language must be 'en' or 'hi' (or empty for auto-detect).")
        groq = self.groq_stt
        model = self.s.groq_stt_model if groq else self.STT_MODEL
        fields = [("model", model)] + ([("response_format", "json")] if groq else []) + ([("language", language)] if language else [])
        body, ctype = encode_multipart(fields, "file", filename, audio, "application/octet-stream")
        url = f"{self.s.groq_base_url}/audio/transcriptions" if groq else f"{self.s.xai_base_url}/stt"
        raw, _ = http_request(url, body, {**self._auth(groq), "Content-Type": ctype}, 60.0)
        try:
            d = json.loads(raw.decode("utf-8"))
            return {"text": (d.get("text") or "").strip(), "language": d.get("language") or language, "duration": d.get("duration")}
        except ValueError:
            raise ProviderError("The speech service returned an unreadable response.")

    def synthesize(self, text, language=None):
        clean = speakable(text, self.s.tts_max_chars)
        if not clean:
            raise ValidationError("There is nothing to read aloud.")
        if not self.tts_enabled:
            raise ProviderUnavailable("Server text-to-speech needs XAI_API_KEY; the browser voice is used instead.")
        lang = language if language in LANGS else detect_language(clean)
        voice = self.s.voice_hi if lang == "hi" else self.s.voice_en
        payload = json.dumps({"text": clean, "voice_id": voice, "language": lang}).encode("utf-8")
        audio, headers = http_request(f"{self.s.xai_base_url}/tts", payload,
                                      {**self._auth(), "Content-Type": "application/json"}, 60.0)
        return audio, (headers.get_content_type() if headers else "audio/mpeg")
