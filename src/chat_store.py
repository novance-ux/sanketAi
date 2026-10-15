"""Durable chat history and long-term conversational memory (SQLite).

Separate from the document index: rebuilding the index never touches conversations.

Ownership: every row belongs to an ``owner_id``, the SHA-256 of a secret user token. The raw
token is never stored. Every read and write filters by owner, so one user cannot open, rename or
delete another user's conversation even with a valid conversation id.

Memory layers handled here:
* persistent history  - conversations and messages (role, content, status, sources, timestamps);
* long-term memory    - per-conversation topic summaries built from the user's own questions
                        (never from model claims), used only as conversational context, with a
                        per-user switch to disable it and controls to view and delete it.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_owner ON conversations (owner_id, updated_at);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    status TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages (conversation_id, id);
CREATE TABLE IF NOT EXISTS user_settings (
    owner_id TEXT PRIMARY KEY,
    memory_enabled INTEGER NOT NULL DEFAULT 1
);
"""

# Assistant message statuses
PENDING = "pending"          # generation started, not finished (stopped or interrupted if left so)
VERIFIED = "verified"        # every documented claim checked against the cited passages
PARTIAL = "partially_verified"  # unsupported claims were removed; the rest is checked
INSUFFICIENT = "insufficient_evidence"
UNVERIFIED = "unverified"    # checks could not run (verifier unavailable); labelled as such
PASSAGES_ONLY = "passages_only"  # no model available; quoted passages only
ERROR = "error"
CHECKED = "citations_checked"  # fast mode: citations exist and numbers occur in the cited sources
NOT_CHECKED = "not_checked"    # checks switched off
STATUS_LABELS = {
    CHECKED: "Citations checked",
    NOT_CHECKED: "Not checked",
    PENDING: "Interrupted - no answer was completed",
    VERIFIED: "Sources verified",
    PARTIAL: "Partly verified - unsupported statements removed",
    INSUFFICIENT: "Not established from the documents",
    UNVERIFIED: "NOT verified - verification unavailable",
    PASSAGES_ONLY: "Passages only - no model available",
    ERROR: "Error",
}

MAX_TITLE = 80


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_user_token() -> str:
    return secrets.token_urlsafe(24)


def owner_id_for(token: str) -> str:
    if not token or len(token) < 16:
        raise ValueError("A user token of at least 16 characters is required.")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class NotFound(LookupError):
    """Conversation does not exist or belongs to another user (deliberately indistinguishable)."""


class ChatStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------ conversations
    def create_conversation(self, owner_id: str, title: str = "New conversation") -> dict:
        cid, ts = uuid.uuid4().hex, now()
        with closing(self._conn()) as conn:
            conn.execute("INSERT INTO conversations VALUES (?, ?, ?, NULL, ?, ?)",
                         (cid, owner_id, title[:MAX_TITLE], ts, ts))
            conn.commit()
        return self.get_conversation(owner_id, cid)

    def get_conversation(self, owner_id: str, conversation_id: str) -> dict:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT * FROM conversations WHERE id = ? AND owner_id = ?",
                               (conversation_id, owner_id)).fetchone()
        if row is None:
            raise NotFound(conversation_id)
        return dict(row)

    def list_conversations(self, owner_id: str, limit: int = 50) -> list[dict]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
                "FROM conversations c WHERE owner_id = ? ORDER BY updated_at DESC LIMIT ?",
                (owner_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, owner_id: str, conversation_id: str, title: str) -> None:
        title = " ".join((title or "").split())[:MAX_TITLE]
        if not title:
            raise ValueError("Title cannot be empty.")
        with closing(self._conn()) as conn:
            cur = conn.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND owner_id = ?",
                               (title, now(), conversation_id, owner_id))
            conn.commit()
        if cur.rowcount == 0:
            raise NotFound(conversation_id)

    def delete_conversation(self, owner_id: str, conversation_id: str) -> None:
        with closing(self._conn()) as conn:
            cur = conn.execute("DELETE FROM conversations WHERE id = ? AND owner_id = ?", (conversation_id, owner_id))
            conn.commit()
        if cur.rowcount == 0:
            raise NotFound(conversation_id)

    def clear_history(self, owner_id: str) -> int:
        with closing(self._conn()) as conn:
            cur = conn.execute("DELETE FROM conversations WHERE owner_id = ?", (owner_id,))
            conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------ messages
    def add_message(self, owner_id: str, conversation_id: str, role: str, content: str,
                    status: str, meta: dict | None = None) -> int:
        self.get_conversation(owner_id, conversation_id)  # ownership check
        ts = now()
        with closing(self._conn()) as conn:
            cur = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, status, meta, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, role, content, status, json.dumps(meta or {}), ts, ts))
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (ts, conversation_id))
            conn.commit()
            return cur.lastrowid

    def update_message(self, owner_id: str, message_id: int, *, content: str, status: str, meta: dict) -> None:
        with closing(self._conn()) as conn:
            cur = conn.execute(
                "UPDATE messages SET content = ?, status = ?, meta = ?, updated_at = ? WHERE id = ? AND "
                "conversation_id IN (SELECT id FROM conversations WHERE owner_id = ?)",
                (content, status, json.dumps(meta), now(), message_id, owner_id))
            conn.commit()
        if cur.rowcount == 0:
            raise NotFound(str(message_id))

    def messages(self, owner_id: str, conversation_id: str) -> list[dict]:
        self.get_conversation(owner_id, conversation_id)
        with closing(self._conn()) as conn:
            rows = conn.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
                                (conversation_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["meta"] = json.loads(d["meta"] or "{}")
            out.append(d)
        return out

    # ------------------------------------------------------------------ titles and memory
    def autotitle(self, owner_id: str, conversation_id: str, first_question: str) -> None:
        conv = self.get_conversation(owner_id, conversation_id)
        if conv["title"] == "New conversation":
            title = " ".join(first_question.split())
            self.rename_conversation(owner_id, conversation_id,
                                     title if len(title) <= 60 else title[:57].rsplit(" ", 1)[0] + "...")

    def refresh_summary(self, owner_id: str, conversation_id: str) -> None:
        """Topic summary from the user's own questions only (no model claims enter memory)."""
        questions = [m["content"] for m in self.messages(owner_id, conversation_id) if m["role"] == "user"]
        if not questions:
            return
        recent = [" ".join(q.split())[:140] for q in questions[-6:]]
        summary = "Questions asked: " + " | ".join(recent)
        with closing(self._conn()) as conn:
            conn.execute("UPDATE conversations SET summary = ? WHERE id = ? AND owner_id = ?",
                         (summary, conversation_id, owner_id))
            conn.commit()

    def memory_enabled(self, owner_id: str) -> bool:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT memory_enabled FROM user_settings WHERE owner_id = ?", (owner_id,)).fetchone()
        return True if row is None else bool(row["memory_enabled"])

    def set_memory_enabled(self, owner_id: str, enabled: bool) -> None:
        with closing(self._conn()) as conn:
            conn.execute("INSERT INTO user_settings VALUES (?, ?) ON CONFLICT(owner_id) DO UPDATE SET "
                         "memory_enabled = excluded.memory_enabled", (owner_id, int(enabled)))
            conn.commit()

    def memories(self, owner_id: str) -> list[dict]:
        return [{"conversation_id": c["id"], "title": c["title"], "summary": c["summary"], "updated_at": c["updated_at"]}
                for c in self.list_conversations(owner_id, 200) if c["summary"]]

    def delete_memories(self, owner_id: str) -> None:
        with closing(self._conn()) as conn:
            conn.execute("UPDATE conversations SET summary = NULL WHERE owner_id = ?", (owner_id,))
            conn.commit()

    def relevant_memories(self, owner_id: str, question: str, exclude_conversation: str | None,
                          limit: int = 2, min_overlap: float = 0.25) -> list[dict]:
        """Topic notes from the user's other conversations that share content words with the question."""
        if not self.memory_enabled(owner_id):
            return []
        q_words = _content_words(question)
        if not q_words:
            return []
        scored = []
        for m in self.memories(owner_id):
            if m["conversation_id"] == exclude_conversation:
                continue
            overlap = len(q_words & _content_words(m["summary"])) / len(q_words)
            if overlap >= min_overlap:
                scored.append((overlap, m))
        scored.sort(key=lambda t: -t[0])
        return [m for _, m in scored[:limit]]


_MEMORY_STOP = set("""about after again also asked before between could does from have here into just know like
more most much questions remind should tell than that their them then there these they this those what when where
which while with would your earlier previous said told mention mentioned discuss discussed last time please explain
""".split())


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", (text or "").lower()) if w not in _MEMORY_STOP}
