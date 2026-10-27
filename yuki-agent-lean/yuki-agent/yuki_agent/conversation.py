"""Conversation manager: per-session history and selected-document memory (SQLite)."""
from __future__ import annotations

from .errors import PermissionDenied


class ConversationManager:
    def __init__(self, db, settings):
        self.db, self.settings = db, settings

    def get_or_create(self, user_id, session_id, title_hint="New chat", project_id=None):
        if session_id:
            s = self.db.get_session(user_id, session_id)
            if not s:
                raise PermissionDenied("Session not found.")
            return s
        return self.db.create_session(user_id, title_hint[:60], project_id)

    def history(self, user_id, session_id):
        msgs = [m for m in self.db.get_messages(user_id, session_id) if m["role"] in ("user", "assistant")
                and not m["metadata"].get("error")]
        return [{"role": m["role"], "content": m["content"]} for m in msgs[-self.settings.history_messages:]]

    def last_assistant(self, user_id, session_id):
        for m in reversed(self.db.get_messages(user_id, session_id)):
            if m["role"] == "assistant" and not m["metadata"].get("error") and m["metadata"].get("saveable", True):
                return m
        return None

    def add(self, session_id, role, content, metadata=None):
        self.db.add_message(session_id, role, content, metadata)
