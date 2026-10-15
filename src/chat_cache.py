"""Cache for retrieval results and verified answers. Performance only; never the source of truth.

Backends: Redis when SANKET_REDIS_URL is set and reachable, otherwise a local SQLite file.

Correctness rules:
* Every key includes the document version (the ingestion fingerprint of both PDFs plus the pipeline
  version), so any change to either PDF or to the index makes all earlier entries unreachable.
  Entries for older versions are purged when the cache is opened.
* Answer keys also include the standalone question, the scope, the model chain and the grounding
  settings. Answers that depended on conversation context are not cached.
* A cached answer is only reused after its cited passages are confirmed to still exist in the
  current index (checked by the caller).
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path

log = logging.getLogger(__name__)


def make_key(kind: str, doc_version: str, *parts: object) -> str:
    raw = json.dumps([kind, doc_version, *parts], sort_keys=True, default=str, ensure_ascii=False)
    return f"sanket:{kind}:{doc_version[:12]}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


class SqliteCache:
    backend = "sqlite"

    def __init__(self, path: Path, doc_version: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.doc_version = doc_version
        with closing(sqlite3.connect(self.path, timeout=10)) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, doc_version TEXT NOT NULL, "
                         "value TEXT NOT NULL, expires_at REAL NOT NULL)")
            # invalidation: anything built from another version of the documents is dropped
            conn.execute("DELETE FROM cache WHERE doc_version != ? OR expires_at < ?", (doc_version, time.time()))
            conn.commit()

    def get(self, key: str):
        with closing(sqlite3.connect(self.path, timeout=10)) as conn:
            row = conn.execute("SELECT value, expires_at, doc_version FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None or row[1] < time.time() or row[2] != self.doc_version:
            return None
        return json.loads(row[0])

    def set(self, key: str, value, ttl_s: int) -> None:
        with closing(sqlite3.connect(self.path, timeout=10)) as conn:
            conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?)",
                         (key, self.doc_version, json.dumps(value, default=str), time.time() + ttl_s))
            conn.commit()

    def delete(self, key: str) -> None:
        with closing(sqlite3.connect(self.path, timeout=10)) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear(self) -> None:
        with closing(sqlite3.connect(self.path, timeout=10)) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()

    def size(self) -> int:
        with closing(sqlite3.connect(self.path, timeout=10)) as conn:
            return conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]


class RedisCache:
    backend = "redis"

    def __init__(self, url: str, doc_version: str):
        import redis

        self.client = redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
        self.client.ping()
        self.doc_version = doc_version
        # invalidation: keys embed the version prefix; drop every other version's keys
        current = f"sanket:*:{doc_version[:12]}:*"
        for key in self.client.scan_iter("sanket:*"):
            if not self._matches(key.decode() if isinstance(key, bytes) else key, current):
                self.client.delete(key)

    @staticmethod
    def _matches(key: str, pattern: str) -> bool:
        return key.split(":")[2:3] == pattern.split(":")[2:3]

    def get(self, key: str):
        raw = self.client.get(key)
        return json.loads(raw) if raw else None

    def set(self, key: str, value, ttl_s: int) -> None:
        self.client.set(key, json.dumps(value, default=str), ex=ttl_s)

    def delete(self, key: str) -> None:
        self.client.delete(key)

    def clear(self) -> None:
        for key in self.client.scan_iter("sanket:*"):
            self.client.delete(key)

    def size(self) -> int:
        return sum(1 for _ in self.client.scan_iter("sanket:*"))


def open_cache(redis_url: str, sqlite_path: Path, doc_version: str):
    if redis_url:
        try:
            return RedisCache(redis_url, doc_version)
        except Exception as exc:  # missing package, server down, bad URL
            log.warning("Redis cache unavailable (%s); using the local SQLite cache.", type(exc).__name__)
    return SqliteCache(sqlite_path, doc_version)
