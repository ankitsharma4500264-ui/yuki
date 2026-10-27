"""Stateless anonymous-user tokens, so a public deployment isolates each visitor's documents.
Not a login system: clearing browser storage loses access. Swap in real auth (Clerk/Auth0/Supabase) for accounts."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time


def _b64(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def new_anonymous_user():
    return "anon_" + secrets.token_hex(8)


def sign_token(secret, user_id, days=180, now=None):
    exp = int((now or time.time()) + days * 86400)
    payload = f"{user_id}.{exp}".encode()
    sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"{_b64(payload)}.{sig}"


def verify_token(secret, token, now=None):
    """Return the user id for a valid, unexpired token, else None."""
    try:
        p64, sig = token.split(".", 1)
        payload = _unb64(p64)
        good = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, good):
            return None
        uid, exp = payload.decode().rsplit(".", 1)
        if int(exp) < (now or time.time()):
            return None
        return uid if uid.startswith("anon_") else None
    except (ValueError, UnicodeDecodeError, AttributeError):
        return None
