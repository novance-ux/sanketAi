"""Retrieval tools 1-5 and evidence-bundle assembly.

Order of authority:
1. exact structured lookup by ``mine_id`` (SQLite) decides which records belong to a mine;
2. keyword (BM25) and semantic (vector) search only *rank* chunks inside that mine;
3. results are de-duplicated and returned with their source references.

Fuzzy matching is used solely to help a person find a mine in the directory. It never
attaches a record to a mine.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rapidfuzz import fuzz

from .config import Settings, get_settings
from .data_normalizer import norm_name
from .database import Database
from .schemas import (
    ContextItem,
    ContextResult,
    EvidenceBundle,
    Incident,
    Mine,
    MineMatch,
    MineSearchResult,
    PreventiveFinding,
    SourceReference,
)
from .vector_store import VectorStore, VectorStoreUnavailable

log = logging.getLogger(__name__)

NO_GEOLOGY_NOTE = (
    "No geological, hydrogeological or mine-plan information is available to this system for "
    "this mine. The SANKET volumes do not contain it and no verified external source is configured."
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    mine_id: str
    incident_id: str
    source_id: str
    chunk_type: str
    text: str
    score: float
    matched_by: list[str]
    source: SourceReference | None = None


class Retriever:
    def __init__(self, db: Database, vectors: VectorStore | None = None, settings: Settings | None = None):
        self.db = db
        self.vectors = vectors
        self.settings = settings or get_settings()
        self.vector_error: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Retriever":
        settings = settings or get_settings()
        return cls(Database(settings.db_path),
                   VectorStore(settings.vector_dir, settings.embedding_backend), settings)

    # ------------------------------------------------------------------ tool 1
    def search_mines(
        self,
        query: str = "",
        category: str | None = None,
        *,
        state: str | None = None,
        district: str | None = None,
        water_only: bool = False,
        limit: int | None = None,
    ) -> MineSearchResult:
        """Find mines by name or documented location. Exact filters run in SQL."""
        candidates = self.db.list_mines(category, state=state, district=district, water_only=water_only)
        q = norm_name(query)
        scored: list[tuple[float, str, Mine]] = []
        for mine in candidates:
            if not q:
                scored.append((0.0, "listing", mine))
                continue
            names = [norm_name(mine.mine_name), *(norm_name(v) for v in mine.name_variants)]
            place = " ".join(norm_name(x) for x in (mine.district, mine.state, mine.operator) if x)
            best: tuple[float, str] | None = None
            for name in names:
                if name == q:
                    cand = (100.0, "exact")
                elif name.startswith(q):
                    cand = (95.0, "prefix")
                elif q in name or set(q.split()) <= set(name.split()):
                    cand = (90.0, "substring")
                else:
                    # tolerate typos, but only against the whole name or a same-length part of it
                    ratio = max(fuzz.ratio(q, name), fuzz.token_sort_ratio(q, name))
                    if len(q) >= 5 and len(name) > len(q):
                        partial = fuzz.partial_ratio(q, name)
                        ratio = max(ratio, partial if partial >= 90 else 0)
                    cand = (float(ratio) * 0.85, "fuzzy") if ratio >= 80 else None
                if cand and (best is None or cand[0] > best[0]):
                    best = cand
            if (best is None or best[0] < 85) and place and (q in place or set(q.split()) <= set(place.split())):
                best = (85.0, "location")
            if best:
                scored.append((best[0], best[1], mine))
        scored.sort(key=lambda t: (-t[0], -t[2].water_incident_count, t[2].mine_name.upper()))
        if limit:
            scored = scored[:limit]
        matches = [
            MineMatch(mine_id=m.mine_id, mine_name=m.mine_name, mine_category=m.mine_category,
                      state=m.state, district=m.district, operator=m.operator,
                      water_incident_count=m.water_incident_count, incident_count=m.incident_count,
                      match_type=kind, score=score)
            for score, kind, m in scored
        ]
        exact = [m for m in matches if m.match_type == "exact"]
        ambiguous = bool(q) and (len(exact) > 1 or (not exact and len(matches) > 1))
        if not matches:
            message = (
                f"No mine matching '{query}' was found in the processed SANKET records"
                + (" for this category" if category else "")
                + ". This means no record was extracted for that name; it does not mean the mine "
                "has no accident history."
            ) if q else "No mines are available for the selected filters."
        elif ambiguous:
            message = (f"{len(matches)} mines match '{query}'. Choose the correct one using the "
                       "district, state and owner shown.")
        else:
            message = f"{len(matches)} mine(s) found."
        return MineSearchResult(query=query, category=category, matches=matches,
                                ambiguous=ambiguous, message=message)

    # ------------------------------------------------------------------ tool 2
    def get_mine_record(self, mine_id: str) -> dict | None:
        mine = self.db.get_mine(mine_id)
        if mine is None:
            return None
        incidents = self.db.incidents_for_mine(mine_id)
        source_ids = [s for i in incidents for s in i.source_ids]
        sources = self.db.get_sources(source_ids)
        return {
            "mine": mine,
            "incident_ids": [i.incident_id for i in incidents],
            "source_ids": list(sources),
            "source_documents": sorted({(s.source_filename, s.page_start, s.page_end) for s in sources.values()}),
        }

    # ------------------------------------------------------------------ tool 3
    def get_inundation_history(
        self,
        mine_id: str,
        *,
        year_from: int | None = None,
        year_to: int | None = None,
        incident_types: list[str] | None = None,
        include_non_water: bool = False,
    ) -> list[Incident]:
        """Incidents for exactly this mine_id. ``incident_types`` filters water relevance."""
        return self.db.incidents_for_mine(
            mine_id, water_only=not include_non_water, relevance=incident_types,
            year_from=year_from, year_to=year_to,
        )

    # ------------------------------------------------------------------ tool 4
    def get_preventive_findings(self, mine_id: str, incident_ids: list[str] | None = None) -> list[PreventiveFinding]:
        return self.db.findings_for_mine(mine_id, incident_ids)

    # ------------------------------------------------------------------ tool 5
    def get_context(self, mine_id: str) -> ContextResult:
        """Verified context only. External items need provenance and a retrieval date."""
        result = ContextResult()
        incidents = self.db.incidents_for_mine(mine_id, water_only=True)
        sources = self.db.get_sources(s for i in incidents for s in i.source_ids)
        for inc in incidents:
            if inc.regulations_cited:
                src = sources.get(inc.source_ids[0])
                result.items.append(ContextItem(
                    title=f"Statutory provisions cited by DGMS for the accident of {inc.date_text}",
                    text="; ".join(inc.regulations_cited),
                    provenance=src.citation() if src else "SANKET volume",
                    retrieved_on=None, kind="in_corpus",
                ))
        external = self._load_external_context(mine_id, result)
        result.items += external
        if not any(i.kind == "external" for i in result.items):
            result.limitations.append(NO_GEOLOGY_NOTE)
        result.limitations.append(
            "The text of the regulations, DGMS circulars and the mine's approved emergency plan "
            "is not held by this system; only the references printed in the SANKET records are shown."
        )
        return result

    def _load_external_context(self, mine_id: str, result: ContextResult) -> list[ContextItem]:
        directory: Path = self.settings.context_dir
        items: list[ContextItem] = []
        if not directory.is_dir():
            return items
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                result.limitations.append(f"Context file {path.name} could not be read ({exc}); ignored.")
                continue
            for entry in data if isinstance(data, list) else [data]:
                applies = entry.get("mine_ids") or []
                if applies != "all" and mine_id not in applies:
                    continue
                missing = [k for k in ("title", "text", "provenance", "retrieved_on") if not entry.get(k)]
                if missing:
                    result.limitations.append(
                        f"An entry in {path.name} was ignored because it lacks: {', '.join(missing)}."
                    )
                    continue
                try:
                    date.fromisoformat(str(entry["retrieved_on"]))
                except ValueError:
                    result.limitations.append(
                        f"An entry in {path.name} was ignored: retrieved_on is not an ISO date."
                    )
                    continue
                items.append(ContextItem(title=entry["title"], text=entry["text"],
                                         provenance=entry["provenance"],
                                         retrieved_on=str(entry["retrieved_on"]), kind="external"))
        return items

    # ------------------------------------------------------------------ hybrid ranking
    def hybrid_search(self, mine_id: str, query: str, limit: int = 8) -> list[RetrievedChunk]:
        """Rank this mine's chunks by keyword + semantic relevance (reciprocal rank fusion)."""
        chunks = {c["chunk_id"]: c for c in self.db.chunks_for_mine(mine_id)}
        if not chunks:
            return []
        ranks: dict[str, float] = {}
        matched: dict[str, list[str]] = {}

        def fuse(ids: list[str], label: str) -> None:
            for rank, cid in enumerate(ids):
                if cid in chunks:  # structured membership check: other mines can never enter
                    ranks[cid] = ranks.get(cid, 0.0) + 1.0 / (60 + rank)
                    matched.setdefault(cid, []).append(label)

        fuse([r["chunk_id"] for r in self.db.keyword_search(mine_id, query, limit * 3)], "keyword")
        if self.vectors is not None and self.vectors.enabled and query.strip():
            try:
                fuse([r["chunk_id"] for r in self.vectors.query(mine_id, query, limit * 3)], "semantic")
                self.vector_error = None
            except VectorStoreUnavailable as exc:
                self.vector_error = str(exc)
                log.debug("Semantic search unavailable: %s", exc)
        ordered = sorted(ranks, key=lambda c: -ranks[c])[:limit]
        sources = self.db.get_sources(chunks[c]["source_id"] for c in ordered)
        return [
            RetrievedChunk(chunk_id=c, mine_id=chunks[c]["mine_id"], incident_id=chunks[c]["incident_id"],
                           source_id=chunks[c]["source_id"], chunk_type=chunks[c]["chunk_type"],
                           text=chunks[c]["text"], score=ranks[c], matched_by=matched[c],
                           source=sources.get(chunks[c]["source_id"]))
            for c in ordered
        ]

    # ------------------------------------------------------------------ evidence bundle
    def build_evidence_bundle(self, mine_id: str) -> EvidenceBundle | None:
        """Steps 1-6 of the orchestration sequence, all keyed on the exact mine_id."""
        mine = self.db.get_mine(mine_id)
        if mine is None:
            return None
        everything = self.db.incidents_for_mine(mine_id)
        water = [i for i in everything if i.water_relevance != "none"]
        other = [i for i in everything if i.water_relevance == "none"]
        findings = self.db.findings_for_mine(mine_id, [i.incident_id for i in water])
        sources = self.db.get_sources(s for i in everything for s in i.source_ids)
        context = self.get_context(mine_id)

        gaps = [NO_GEOLOGY_NOTE] if not any(i.kind == "external" for i in context.items) else []
        if mine.identity_basis == "name_only":
            gaps.append(
                "The source identifies this mine by name only (historic tables). Its state, "
                "district and owner are not documented, so confirm it is the mine you intend."
            )
        if water and not any(i.description for i in water):
            gaps.append(
                "Only table entries exist for this mine's water-related accidents (date, casualties, "
                "cause). The source gives no narrative, mechanism, affected workings or findings."
            )
        if water and not findings:
            gaps.append("No DGMS findings on preventable causes are documented for these accidents.")
        if any(i.incident_date is None for i in water):
            gaps.append("At least one accident date is ambiguous in the source; only the year is reliable.")
        if mine.possibly_related_mine_ids:
            gaps.append(
                f"{len(mine.possibly_related_mine_ids)} other record group(s) carry the same mine name "
                "but could not be confirmed as this mine (location not printed). They are NOT used here."
            )
        gaps += [
            "Measures actually implemented after these accidents, and whether any are in place or "
            "effective today, are not documented.",
            "Current mine status, present water levels, pumping capacity and live monitoring data "
            "are not available to this system.",
            "The yearly narratives cover fatal accidents from 2016 to 2022 only; non-fatal inundations "
            "and dangerous occurrences at this mine are not individually recorded.",
        ]
        return EvidenceBundle(mine=mine, incidents=water, other_incidents=other, findings=findings,
                              sources=sources, context=context, data_gaps=gaps)
