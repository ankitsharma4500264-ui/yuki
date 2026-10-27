"""Safety & validation layer: input checks, prompt-injection handling, rate limits."""
from __future__ import annotations

import os
import re
import threading
import time
from collections import defaultdict, deque

from .errors import ValidationError

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |any |the )?(previous|prior|above|system)",
    r"(reveal|show|print|repeat|output) (your|the) (system )?(prompt|instructions|rules)",
    r"\byou are now\b",
    r"\bnew instructions?\s*:",
    r"\bsystem prompt\b",
    r"act as (an? )?(unrestricted|jailbroken)",
    r"do anything now",
    r"<\s*/?\s*(system|assistant|evidence)\s*>",
    r"(delete|erase|remove) (all )?(my |the )?(documents|notes|files)",
]
_INJ = [re.compile(p, re.I) for p in INJECTION_PATTERNS]
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def validate_message(message, max_chars):
    if not isinstance(message, str):
        raise ValidationError("Message must be text.")
    msg = _CTRL.sub("", message).strip()
    if not msg:
        raise ValidationError("Message is empty.")
    if len(msg) > max_chars:
        raise ValidationError(f"Message is too long (max {max_chars} characters).")
    return msg


def detect_injection(text):
    """Return the patterns that matched (used to flag, never to obey)."""
    return [p.pattern for p in _INJ if p.search(text or "")]


def sanitize_evidence(text):
    """Neutralise anything in document text that could fake structure or citations."""
    text = re.sub(r"<\s*/?\s*evidence\s*>", "", text, flags=re.I)
    text = re.sub(r"\[(S\d+)\]", r"(\1)", text)
    return text.replace('"""', "'''")


def safe_filename(name):
    base = os.path.basename((name or "").replace("\\", "/"))
    base = re.sub(r"[^\w.\- ]", "_", base).strip(" .")
    return (base or "document")[:120]


def validate_upload(filename, size, head, max_mb):
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError("Unsupported file type. Allowed: PDF, TXT, MD.")
    if size <= 0:
        raise ValidationError("File is empty.")
    if size > max_mb * 1024 * 1024:
        raise ValidationError(f"File exceeds the {max_mb} MB limit.")
    if ext == ".pdf" and not head.startswith(b"%PDF"):
        raise ValidationError("File does not look like a valid PDF.")
    if ext in (".txt", ".md") and b"\x00" in head:
        raise ValidationError("File does not look like plain text.")
    return ext


class RateLimiter:
    def __init__(self, per_minute):
        self.per_minute = per_minute
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key):
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True
