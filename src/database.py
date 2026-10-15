"""SQLite storage for the normalised records. This is the source of truth for mine identity,
incident membership and counts; the vector index is only ever a ranking aid on top of it.

Each row keeps the full validated Pydantic model as JSON plus the columns needed for exact
filtering. Connections are short-lived so the Streamlit app never blocks an index rebuild.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from .schemas import Incident, Mine, PreventiveFinding, SourceReference

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE documents (
    filename TEXT PRIMARY KEY, sha256 TEXT NOT NULL, page_count INTEGER NOT NULL,
    category TEXT, title TEXT, records INTEGER NOT NULL
);
CREATE TABLE mines (
    mine_id TEXT PRIMARY KEY, mine_name TEXT NOT NULL, name_key TEXT NOT NULL,
    mine_category TEXT NOT NULL CHECK (mine_category IN ('coal', 'metal_non_coal')),
    state TEXT, district TEXT, operator TEXT, identity_basis TEXT NOT NULL,
    incident_count INTEGER NOT NULL, water_incident_count INTEGER NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX idx_mines_category ON mines (mine_category, water_incident_count);
CREATE TABLE incidents (
    incident_id TEXT PRIMARY KEY,
    mine_id TEXT NOT NULL REFERENCES mines (mine_id),
    incident_year INTEGER, incident_date TEXT, water_relevance TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX idx_incidents_mine ON incidents (mine_id, incident_year);
CREATE TABLE findings (
    measure_id TEXT PRIMARY KEY,
    mine_id TEXT NOT NULL REFERENCES mines (mine_id),
    incident_id TEXT NOT NULL REFERENCES incidents (incident_id),
    measure_category TEXT, data TEXT NOT NULL
);
CREATE INDEX idx_findings_mine ON findings (mine_id, incident_id);
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY, source_filename TEXT NOT NULL, page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL, section TEXT NOT NULL, verification_status TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE incident_sources (
    incident_id TEXT NOT NULL REFERENCES incidents (incident_id),
    source_id TEXT NOT NULL REFERENCES sources (source_id),
    PRIMARY KEY (incident_id, source_id)
);
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY, mine_id TEXT NOT NULL, incident_id TEXT NOT NULL,
    source_id TEXT NOT NULL, chunk_type TEXT NOT NULL, text TEXT NOT NULL
);
CREATE INDEX idx_chunks_mine ON chunks (mine_id);
CREATE VIRTUAL TABLE chunks_fts USING fts5 (chunk_id UNINDEXED, mine_id UNINDEXED, text, tokenize = 'porter unicode61');
CREATE TABLE page_chunks (
    chunk_id TEXT PRIMARY KEY, source_filename TEXT NOT NULL, category TEXT,
    page INTEGER NOT NULL, text TEXT NOT NULL
);
CREATE INDEX idx_page_chunks_page ON page_chunks (source_filename, page);
CREATE VIRTUAL TABLE pages_fts USING fts5 (chunk_id UNINDEXED, text, tokenize = 'porter unicode61');
"""

# Words that match nearly every page and only dilute keyword ranking.
_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how i in is it its many me much of on or "
    "per show tell than that the their there these this to was were what when where which who why will "
    "with you your about give list all any".split()
)


class DatabaseNotReady(RuntimeError):
    """The structured database has not been built yet (run the ingestion command)."""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    # ------------------------------------------------------------------ connections
    def exists(self) -> bool:
        return self.path.is_file() and self.path.stat().st_size > 0

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if not self.exists():
            raise DatabaseNotReady(
                f"No database at {self.path}. Run `python -m src.ingestion` to build it."
            )
        uri = f"{self.path.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            yield conn

    # ------------------------------------------------------------------ writing
    def write(
        self,
        *,
        mines: Iterable[Mine],
        incidents: Iterable[Incident],
        findings: Iterable[PreventiveFinding],
        sources: Iterable[SourceReference],
        chunks: Iterable[dict],
        documents: Iterable[dict],
        meta: dict[str, str],
        page_chunks: Iterable[dict] = (),
    ) -> None:
        """Build a fresh database next to the old one and swap it in atomically."""
        from .data_normalizer import norm_name

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.unlink(missing_ok=True)
        with closing(sqlite3.connect(tmp)) as conn:
            conn.executescript(SCHEMA)
            conn.executemany("INSERT INTO meta VALUES (?, ?)", list(meta.items()))
            conn.executemany(
                "INSERT INTO documents VALUES (:filename, :sha256, :page_count, :category, :title, :records)",
                list(documents),
            )
            conn.executemany(
                "INSERT INTO mines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(m.mine_id, m.mine_name, norm_name(m.mine_name), m.mine_category, m.state,
                  m.district, m.operator, m.identity_basis, m.incident_count,
                  m.water_incident_count, m.model_dump_json()) for m in mines],
            )
            incidents = list(incidents)
            conn.executemany(
                "INSERT INTO incidents VALUES (?, ?, ?, ?, ?, ?)",
                [(i.incident_id, i.mine_id, i.incident_year, i.incident_date, i.water_relevance,
                  i.model_dump_json()) for i in incidents],
            )
            conn.executemany(
                "INSERT INTO findings VALUES (?, ?, ?, ?, ?)",
                [(f.measure_id, f.mine_id, f.incident_id, f.measure_category, f.model_dump_json())
                 for f in findings],
            )
            conn.executemany(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(s.source_id, s.source_filename, s.page_start, s.page_end, s.section,
                  s.verification_status, s.model_dump_json()) for s in sources],
            )
            conn.executemany(
                "INSERT INTO incident_sources VALUES (?, ?)",
                [(i.incident_id, s) for i in incidents for s in i.source_ids],
            )
            chunks = list(chunks)
            conn.executemany(
                "INSERT INTO chunks VALUES (:chunk_id, :mine_id, :incident_id, :source_id, :chunk_type, :text)",
                chunks,
            )
            conn.executemany(
                "INSERT INTO chunks_fts (chunk_id, mine_id, text) VALUES (:chunk_id, :mine_id, :text)",
                chunks,
            )
            page_chunks = list(page_chunks)
            conn.executemany(
                "INSERT INTO page_chunks VALUES (:chunk_id, :source_filename, :category, :page, :text)",
                page_chunks,
            )
            conn.executemany(
                "INSERT INTO pages_fts (chunk_id, text) VALUES (:chunk_id, :text)", page_chunks
            )
            conn.commit()
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ reading
    def meta(self) -> dict[str, str]:
        with self._connect() as conn:
            return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}

    def documents(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM documents ORDER BY filename")]

    def category_counts(self) -> dict[str, dict[str, int]]:
        """Mines per category, computed from stored records (never hardcoded)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT mine_category, COUNT(*) AS mines, "
                "SUM(water_incident_count > 0) AS water_mines, "
                "SUM(incident_count) AS incidents, SUM(water_incident_count) AS water_incidents "
                "FROM mines GROUP BY mine_category"
            ).fetchall()
        return {r["mine_category"]: {k: r[k] or 0 for k in r.keys() if k != "mine_category"} for r in rows}

    def list_mines(
        self,
        category: str | None = None,
        *,
        state: str | None = None,
        district: str | None = None,
        water_only: bool = False,
    ) -> list[Mine]:
        sql, args = "SELECT data FROM mines WHERE 1 = 1", []
        if category:
            sql += " AND mine_category = ?"
            args.append(category)
        if state:
            sql += " AND state = ? COLLATE NOCASE"
            args.append(state)
        if district:
            sql += " AND district = ? COLLATE NOCASE"
            args.append(district)
        if water_only:
            sql += " AND water_incident_count > 0"
        sql += " ORDER BY mine_name COLLATE NOCASE, state, district"
        with self._connect() as conn:
            return [Mine.model_validate_json(r["data"]) for r in conn.execute(sql, args)]

    def get_mine(self, mine_id: str) -> Mine | None:
        with self._connect() as conn:
            row = conn.execute("SELECT data FROM mines WHERE mine_id = ?", (mine_id,)).fetchone()
        return Mine.model_validate_json(row["data"]) if row else None

    def locations(self, category: str | None = None) -> dict[str, list[str]]:
        """Distinct documented states -> districts, for filter widgets."""
        sql, args = "SELECT DISTINCT state, district FROM mines WHERE state IS NOT NULL", []
        if category:
            sql += " AND mine_category = ?"
            args.append(category)
        out: dict[str, set[str]] = {}
        with self._connect() as conn:
            for r in conn.execute(sql, args):
                out.setdefault(r["state"], set()).add(r["district"] or "")
        return {s: sorted(d for d in ds if d) for s, ds in sorted(out.items())}

    def incidents_for_mine(
        self,
        mine_id: str,
        *,
        water_only: bool = False,
        relevance: Iterable[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[Incident]:
        """Exact structured lookup: only rows whose mine_id equals the requested one."""
        sql, args = "SELECT data FROM incidents WHERE mine_id = ?", [mine_id]
        if relevance:
            values = list(relevance)
            sql += f" AND water_relevance IN ({','.join('?' * len(values))})"
            args += values
        elif water_only:
            sql += " AND water_relevance != 'none'"
        if year_from is not None:
            sql += " AND incident_year >= ?"
            args.append(year_from)
        if year_to is not None:
            sql += " AND incident_year <= ?"
            args.append(year_to)
        sql += " ORDER BY incident_year, incident_date, incident_id"
        with self._connect() as conn:
            return [Incident.model_validate_json(r["data"]) for r in conn.execute(sql, args)]

    def findings_for_mine(self, mine_id: str, incident_ids: Iterable[str] | None = None) -> list[PreventiveFinding]:
        sql, args = "SELECT data FROM findings WHERE mine_id = ?", [mine_id]
        if incident_ids is not None:
            values = list(incident_ids)
            if not values:
                return []
            sql += f" AND incident_id IN ({','.join('?' * len(values))})"
            args += values
        sql += " ORDER BY measure_id"
        with self._connect() as conn:
            return [PreventiveFinding.model_validate_json(r["data"]) for r in conn.execute(sql, args)]

    def get_sources(self, source_ids: Iterable[str]) -> dict[str, SourceReference]:
        values = list(dict.fromkeys(source_ids))
        if not values:
            return {}
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT data FROM sources WHERE source_id IN ({','.join('?' * len(values))})", values
            ).fetchall()
        found = {s.source_id: s for s in (SourceReference.model_validate_json(r["data"]) for r in rows)}
        return {sid: found[sid] for sid in values if sid in found}

    def get_mine_for_incident(self, incident_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT mine_id FROM incidents WHERE incident_id = ?", (incident_id,)).fetchone()
        return row["mine_id"] if row else None

    def chunks_for_mine(self, mine_id: str) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM chunks WHERE mine_id = ? ORDER BY chunk_id", (mine_id,))]

    def all_chunks(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM chunks ORDER BY chunk_id")]

    def keyword_search(self, mine_id: str, query: str, limit: int = 10) -> list[dict]:
        """BM25 keyword search restricted to one mine by an exact mine_id filter."""
        terms = [t for t in re.findall(r"[A-Za-z0-9]{2,}", query or "")][:24]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.*, bm25(chunks_fts) AS score FROM chunks_fts "
                "JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id "
                "WHERE chunks_fts MATCH ? AND chunks_fts.mine_id = ? ORDER BY score LIMIT ?",
                (match, mine_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ whole-document pages
    def page_keyword_search(self, query: str, limit: int = 12, category: str | None = None) -> list[dict]:
        """BM25 search over every text page of both volumes (used by the document chat)."""
        terms = [t for t in re.findall(r"[A-Za-z0-9]{2,}", query or "") if t.lower() not in _STOPWORDS][:24]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        sql = ("SELECT p.*, bm25(pages_fts) AS score FROM pages_fts "
               "JOIN page_chunks p ON p.chunk_id = pages_fts.chunk_id WHERE pages_fts MATCH ?")
        args: list = [match]
        if category:
            sql += " AND p.category = ?"
            args.append(category)
        sql += " ORDER BY score LIMIT ?"
        args.append(limit)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, args)]

    def get_page_chunks(self, chunk_ids: Iterable[str]) -> dict[str, dict]:
        values = list(dict.fromkeys(chunk_ids))
        if not values:
            return {}
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM page_chunks WHERE chunk_id IN ({','.join('?' * len(values))})", values
            ).fetchall()
        return {r["chunk_id"]: dict(r) for r in rows}

    def all_page_chunks(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM page_chunks ORDER BY chunk_id")]

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            return {
                t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("mines", "incidents", "findings", "sources", "chunks", "page_chunks")
            }


def dump_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
