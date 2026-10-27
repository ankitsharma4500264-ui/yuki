"""SQLite persistence. Every query is parameterised and scoped by user_id."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from .errors import YukiError

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, filename TEXT NOT NULL, path TEXT,
  pages INTEGER DEFAULT 0, chunks INTEGER DEFAULT 0, status TEXT DEFAULT 'processing',
  error TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id);
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT, selected_docs TEXT DEFAULT '[]',
  project_id TEXT, preferences TEXT DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT DEFAULT '{}', created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_msgs_session ON messages(session_id);
CREATE TABLE IF NOT EXISTS projects(
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notes(
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL, sources TEXT DEFAULT '[]',
  created_at TEXT NOT NULL, UNIQUE(project_id, title));
CREATE TABLE IF NOT EXISTS pending_actions(
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT, tool TEXT NOT NULL,
  arguments TEXT NOT NULL, preview TEXT, status TEXT DEFAULT 'pending', created_at TEXT NOT NULL);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id():
    return uuid.uuid4().hex


class Database:
    def __init__(self, path=":memory:"):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- helpers
    def _run(self, sql, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _all(self, sql, params=()):
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _one(self, sql, params=()):
        rows = self._all(sql, params)
        return rows[0] if rows else None

    def ping(self):
        self._one("SELECT 1 AS ok")
        return True

    # -- documents
    def add_document(self, user_id, filename, path=None, doc_id=None):
        doc_id = doc_id or new_id()
        self._run("INSERT INTO documents(id,user_id,filename,path,created_at) VALUES(?,?,?,?,?)",
                  (doc_id, user_id, filename, path, now()))
        return doc_id

    def set_document_status(self, doc_id, status, pages=0, chunks=0, error=None):
        self._run("UPDATE documents SET status=?,pages=?,chunks=?,error=? WHERE id=?",
                  (status, pages, chunks, error, doc_id))

    def get_document(self, user_id, doc_id):
        return self._one("SELECT * FROM documents WHERE id=? AND user_id=?", (doc_id, user_id))

    def get_documents(self, user_id, ids):
        ids = list(dict.fromkeys(ids))
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        return self._all(f"SELECT * FROM documents WHERE user_id=? AND id IN ({q})", (user_id, *ids))

    def list_documents(self, user_id, ready_only=False):
        sql = "SELECT * FROM documents WHERE user_id=?"
        if ready_only:
            sql += " AND status='ready'"
        return self._all(sql + " ORDER BY created_at DESC", (user_id,))

    def delete_document(self, user_id, doc_id):
        self._run("DELETE FROM documents WHERE id=? AND user_id=?", (doc_id, user_id))

    # -- sessions
    @staticmethod
    def _sess(row):
        if row:
            row["selected_docs"] = json.loads(row["selected_docs"] or "[]")
            row["preferences"] = json.loads(row["preferences"] or "{}")
        return row

    def create_session(self, user_id, title="New chat", project_id=None, selected_docs=None):
        sid = new_id()
        t = now()
        self._run("INSERT INTO sessions(id,user_id,title,selected_docs,project_id,created_at,updated_at) "
                  "VALUES(?,?,?,?,?,?,?)",
                  (sid, user_id, title[:80], json.dumps(selected_docs or []), project_id, t, t))
        return self.get_session(user_id, sid)

    def get_session(self, user_id, sid):
        return self._sess(self._one("SELECT * FROM sessions WHERE id=? AND user_id=?", (sid, user_id)))

    def list_sessions(self, user_id):
        rows = self._all("SELECT id,title,project_id,created_at,updated_at FROM sessions "
                         "WHERE user_id=? ORDER BY updated_at DESC", (user_id,))
        return rows

    def update_session(self, user_id, sid, **fields):
        allowed = {"title", "selected_docs", "project_id", "preferences"}
        sets, vals = [], []
        for k, v in fields.items():
            if k not in allowed:
                raise YukiError(f"Cannot update session field '{k}'.")
            sets.append(f"{k}=?")
            vals.append(json.dumps(v) if k in ("selected_docs", "preferences") else v)
        sets.append("updated_at=?")
        vals.append(now())
        self._run(f"UPDATE sessions SET {','.join(sets)} WHERE id=? AND user_id=?", (*vals, sid, user_id))

    def delete_session(self, user_id, sid):
        self._run("DELETE FROM sessions WHERE id=? AND user_id=?", (sid, user_id))

    def delete_all_sessions(self, user_id):
        self._run("DELETE FROM sessions WHERE user_id=?", (user_id,))

    # -- messages
    def add_message(self, session_id, role, content, metadata=None):
        self._run("INSERT INTO messages(session_id,role,content,metadata,created_at) VALUES(?,?,?,?,?)",
                  (session_id, role, content, json.dumps(metadata or {}), now()))
        self._run("UPDATE sessions SET updated_at=? WHERE id=?", (now(), session_id))

    def get_messages(self, user_id, session_id):
        rows = self._all(
            "SELECT m.id,m.role,m.content,m.metadata,m.created_at FROM messages m "
            "JOIN sessions s ON s.id=m.session_id WHERE s.id=? AND s.user_id=? ORDER BY m.id",
            (session_id, user_id))
        for r in rows:
            r["metadata"] = json.loads(r["metadata"] or "{}")
        return rows

    def delete_last_exchange(self, user_id, session_id):
        """Remove the trailing assistant reply and the user message before it; return that message."""
        msgs = self.get_messages(user_id, session_id)
        ids, user_text = [], None
        for m in reversed(msgs):
            ids.append(m["id"])
            if m["role"] == "user":
                user_text = m["content"]
                break
        if user_text is None:
            return None
        q = ",".join("?" * len(ids))
        self._run(f"DELETE FROM messages WHERE id IN ({q})", tuple(ids))
        return user_text

    # -- projects & notes
    def create_project(self, user_id, name):
        pid = new_id()
        self._run("INSERT INTO projects(id,user_id,name,created_at) VALUES(?,?,?,?)",
                  (pid, user_id, name[:120], now()))
        return self.get_project(user_id, pid)

    def get_project(self, user_id, pid):
        return self._one("SELECT * FROM projects WHERE id=? AND user_id=?", (pid, user_id))

    def list_projects(self, user_id):
        return self._all("SELECT * FROM projects WHERE user_id=? ORDER BY created_at", (user_id,))

    def add_note(self, user_id, project_id, title, content, sources):
        """Insert only. Never overwrites an existing note."""
        if not self.get_project(user_id, project_id):
            raise YukiError("Project not found.")
        nid = new_id()
        try:
            self._run("INSERT INTO notes(id,project_id,user_id,title,content,sources,created_at) "
                      "VALUES(?,?,?,?,?,?,?)",
                      (nid, project_id, user_id, title, content, json.dumps(sources), now()))
        except sqlite3.IntegrityError:
            raise YukiError(f"A note titled '{title}' already exists in this project. "
                            "Existing notes are never overwritten; choose another title.")
        return nid

    def list_notes(self, user_id, project_id):
        rows = self._all("SELECT * FROM notes WHERE user_id=? AND project_id=? ORDER BY created_at",
                         (user_id, project_id))
        for r in rows:
            r["sources"] = json.loads(r["sources"] or "[]")
        return rows

    # -- pending confirmations
    def create_pending(self, user_id, session_id, tool, arguments, preview):
        pid = new_id()
        self._run("INSERT INTO pending_actions(id,user_id,session_id,tool,arguments,preview,created_at) "
                  "VALUES(?,?,?,?,?,?,?)",
                  (pid, user_id, session_id, tool, json.dumps(arguments), preview, now()))
        return pid

    def get_pending(self, user_id, pid):
        row = self._one("SELECT * FROM pending_actions WHERE id=? AND user_id=?", (pid, user_id))
        if row:
            row["arguments"] = json.loads(row["arguments"])
        return row

    def set_pending_status(self, pid, status):
        self._run("UPDATE pending_actions SET status=? WHERE id=?", (status, pid))

    # -- privacy
    def delete_user_data(self, user_id):
        """Delete everything stored for a user. Returns the removed document rows (for file/vector cleanup)."""
        docs = self.list_documents(user_id)
        for table in ("documents", "sessions", "projects", "pending_actions"):
            self._run(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
        return docs
