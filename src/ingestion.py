"""Ingestion pipeline: locate PDFs -> extract -> normalise -> validate -> store -> index -> audit.

Run independently of the UI:

    python -m src.ingestion              # incremental: does nothing if inputs are unchanged
    python -m src.ingestion --rebuild    # force a full rebuild of the database and index
    python -m src.ingestion --skip-vectors

The source PDFs are opened read-only and are never moved, modified or deleted. If the
volumes cannot be found or read the pipeline stops with an explanation; it never falls back
to sample data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .config import PIPELINE_VERSION, Settings, configure_logging, get_settings
from .data_normalizer import NormalizedDataset, load_aliases, normalize
from .database import Database, dump_json
from .pdf_extractor import (
    CATEGORY_COAL,
    CATEGORY_NON_COAL,
    ExtractionResult,
    PdfExtractionError,
    extract_document,
    file_sha256,
)
from .schemas import CATEGORY_LABELS, SECTION_LABELS, WATER_LABELS
from .vector_store import VectorStore, VectorStoreUnavailable

log = logging.getLogger(__name__)

# Facts the two volumes do not contain at mine level. Stated in the audit and in the UI so
# that an absent field is never mistaken for "nothing happened".
FIELDS_ABSENT_FROM_SOURCE = [
    "Geological or hydrogeological description of any mine (strata, aquifers, faults, seam depth).",
    "DGMS region/zone of an individual mine (regions appear only in aggregate statistical tables).",
    "Water level, inflow rate or volume as a recorded field (quantities appear only inside narratives).",
    "Preventive measures actually implemented after an accident, their dates and their effectiveness.",
    "Present-day status of any mine, barrier, pump or other installation.",
    "Narratives for accidents before 2016: the historic tables give only date, mine name, casualties and cause.",
    "Mine-level details of serious (non-fatal) accidents and dangerous occurrences (only yearly counts are published).",
    "State, district or owner for mines that appear only in the historic tables.",
]


class IngestionError(RuntimeError):
    """A blocker that prevents building the dataset (reported to the operator verbatim)."""


def discover_pdfs(settings: Settings) -> list[Path]:
    """Find candidate PDFs in the configured directories (non-recursive, de-duplicated)."""
    found: dict[str, Path] = {}
    for directory in settings.pdf_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.pdf")) + sorted(directory.glob("*.PDF")):
            found.setdefault(str(path.resolve()).lower(), path.resolve())
    return list(found.values())


def build_chunks(ds: NormalizedDataset) -> list[dict]:
    """One 'incident' chunk and (when present) one 'findings' chunk per incident.

    Both carry the same identifying header, so a finding is never separated from the mine
    and accident it belongs to, and every chunk keeps its mine_id / incident_id / source_id.
    """
    mines = {m.mine_id: m for m in ds.mines}
    chunks: list[dict] = []
    for inc in ds.incidents:
        mine = mines[inc.mine_id]
        place = ", ".join(x for x in (mine.district, mine.state) if x) or "location not stated in source"
        header = (
            f"Mine: {inc.mine_name_in_source} ({CATEGORY_LABELS[mine.mine_category]}; {place}). "
            f"Date: {inc.date_text} (year {inc.incident_year}). "
            f"Cause as classified in source: {inc.cause_label or 'not stated'}."
        )
        body = inc.description or (
            f"Table record. Persons killed: {inc.persons_killed_count}; seriously injured: "
            f"{inc.seriously_injured_count}. {inc.court_of_inquiry or ''}"
        )
        base = {"mine_id": inc.mine_id, "incident_id": inc.incident_id, "source_id": inc.source_ids[0]}
        chunks.append({**base, "chunk_id": f"{inc.incident_id}::incident", "chunk_type": "incident",
                       "text": f"{header}\n{body}".strip()})
        if inc.findings_text:
            chunks.append({**base, "chunk_id": f"{inc.incident_id}::findings", "chunk_type": "findings",
                           "text": f"{header}\nDGMS findings on what would have averted the accident:\n"
                                   f"{inc.findings_text}"})
    return chunks


def _fingerprint(pdfs: list[Path], alias_file: Path) -> str:
    h = hashlib.sha256(PIPELINE_VERSION.encode())
    for path in sorted(pdfs, key=lambda p: p.name):
        h.update(path.name.encode())
        h.update(file_sha256(path).encode())
    if alias_file.is_file():
        h.update(alias_file.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- audit report
def build_report(results: list[ExtractionResult], failures: list[dict], ds: NormalizedDataset,
                 vector_status: dict) -> dict:
    sources = {s.source_id: s for s in ds.sources}
    mines = {m.mine_id: m for m in ds.mines}
    docs = []
    for res in results:
        doc = res.document
        parsed = Counter(r.year for r in res.fatal_records)
        docs.append({
            "source_filename": doc.filename,
            "sha256": doc.sha256,
            "volume": doc.title,
            "mine_category": doc.category,
            "pages_processed": doc.page_count,
            "pages_with_selectable_text": sum(p.kind == "text" for p in doc.pages),
            "image_only_pages": doc.image_only_pages,
            "blank_pages": doc.blank_pages,
            "ocr": "Not required: every record-bearing page has a selectable text layer. "
                   "Image-only pages are covers, section dividers and chart graphics.",
            "section_pages": {k: [v[0], v[-1]] if v else None for k, v in res.section_pages.items()},
            "records_extracted": {
                "fatal_accident_summaries": len(res.fatal_records),
                "major_accident_table_rows": len(res.major_rows),
                "court_of_inquiry_rows": len(res.court_rows),
                "total": len(res.fatal_records) + len(res.major_rows) + len(res.court_rows),
            },
            "fatal_accident_cross_check": [
                {"year": y, "stated_in_volume": res.stated_fatal_totals.get(y), "parsed": parsed.get(y, 0),
                 "match": res.stated_fatal_totals.get(y) == parsed.get(y, 0)}
                for y in sorted(set(parsed) | set(res.stated_fatal_totals))
            ],
            "extraction_warnings": res.warnings,
        })

    by_cat = {}
    for cat in (CATEGORY_COAL, CATEGORY_NON_COAL):
        cat_mines = [m for m in ds.mines if m.mine_category == cat]
        cat_inc = [i for i in ds.incidents if mines[i.mine_id].mine_category == cat]
        by_cat[cat] = {
            "label": CATEGORY_LABELS[cat],
            "mines": len(cat_mines),
            "mines_with_water_related_records": sum(m.water_incident_count > 0 for m in cat_mines),
            "mines_identified_by_name_only": sum(m.identity_basis == "name_only" for m in cat_mines),
            "incidents": len(cat_inc),
            "incidents_by_water_relevance": dict(Counter(i.water_relevance for i in cat_inc)),
            "year_range": [min((i.incident_year for i in cat_inc if i.incident_year), default=None),
                           max((i.incident_year for i in cat_inc if i.incident_year), default=None)],
        }

    water_records = []
    for inc in ds.incidents:
        if inc.water_relevance == "none":
            continue
        mine = mines[inc.mine_id]
        water_records.append({
            "incident_id": inc.incident_id, "mine_id": mine.mine_id, "mine_name": inc.mine_name_in_source,
            "mine_category": mine.mine_category, "state": mine.state, "district": mine.district,
            "date_text": inc.date_text, "year": inc.incident_year, "cause_in_source": inc.cause_label,
            "water_relevance": inc.water_relevance, "persons_killed": inc.persons_killed_count,
            "sections": inc.record_sections,
            "source_pages": [{"source_filename": sources[s].source_filename, "pages": sources[s].pages}
                             for s in inc.source_ids],
        })

    ambiguous = [
        {"source_id": s.source_id, "source_filename": s.source_filename, "pages": s.pages,
         "section": s.section, "warnings": s.warnings}
        for s in ds.sources if s.verification_status == "needs_review"
    ]
    n = len(ds.incidents)
    completeness = {
        "incidents_total": n,
        "incidents_without_exact_date": sum(i.incident_date is None for i in ds.incidents),
        "incidents_without_narrative": sum(i.description is None for i in ds.incidents),
        "incidents_without_findings_block": sum(i.findings_text is None for i in ds.incidents),
        "incidents_without_casualty_count": sum(i.persons_killed_count is None for i in ds.incidents),
        "mines_total": len(ds.mines),
        "mines_without_state_or_district": sum(m.state is None for m in ds.mines),
        "mines_without_operator": sum(m.operator is None for m in ds.mines),
        "mines_with_geology_summary": sum(m.geology_summary is not None for m in ds.mines),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_version": PIPELINE_VERSION,
        "documents": docs,
        "unreadable_documents": failures,
        "totals": {
            "mines": len(ds.mines), "incidents": n, "preventive_findings": len(ds.findings),
            "source_references": len(ds.sources),
            "water_related_incidents": len(water_records),
            "raw_records_extracted": sum(d["records_extracted"]["total"] for d in docs),
        },
        "by_category": by_cat,
        "water_related_records": water_records,
        "record_merges": ds.merges,
        "aliases_applied": ds.aliases_applied,
        "likely_duplicates_not_merged": ds.duplicate_candidates,
        "ambiguous_or_flagged_records": ambiguous,
        "field_completeness": completeness,
        "fields_absent_from_source": FIELDS_ABSENT_FROM_SOURCE,
        "schema_validation_warnings": ds.warnings,
        "vector_index": vector_status,
    }


def render_report_markdown(report: dict) -> str:
    out = ["# SANKET AI - Data Extraction Audit", "",
           f"Generated: {report['generated_at']} | pipeline version {report['pipeline_version']}", ""]
    t = report["totals"]
    out += ["## Summary", "",
            f"- Raw records extracted from the PDFs: **{t['raw_records_extracted']}**",
            f"- Normalised incidents: **{t['incidents']}** (of which water-related: **{t['water_related_incidents']}**)",
            f"- Mines identified: **{t['mines']}**",
            f"- Documented preventive findings (DGMS 'Had ... been ...' items): **{t['preventive_findings']}**",
            f"- Source references with page provenance: **{t['source_references']}**", ""]
    for d in report["documents"]:
        r = d["records_extracted"]
        out += [f"## Source: `{d['source_filename']}`", "",
                f"- Volume: {d['volume']} (category: {d['mine_category']})",
                f"- Pages processed: {d['pages_processed']} "
                f"({d['pages_with_selectable_text']} with selectable text, "
                f"{len(d['image_only_pages'])} image-only, {len(d['blank_pages'])} blank)",
                f"- Image-only pages: {d['image_only_pages']}",
                f"- OCR: {d['ocr']}",
                f"- Records extracted: {r['total']} = {r['fatal_accident_summaries']} fatal-accident summaries "
                f"+ {r['major_accident_table_rows']} major-accident table rows "
                f"+ {r['court_of_inquiry_rows']} court-of-inquiry rows",
                "- Section page ranges: " + "; ".join(
                    f"{k} pp. {v[0]}-{v[1]}" for k, v in d["section_pages"].items() if v), "",
                "| Year | Fatal accidents stated in volume | Records parsed | Match |", "|---|---|---|---|"]
        out += [f"| {c['year']} | {c['stated_in_volume']} | {c['parsed']} | {'yes' if c['match'] else 'NO'} |"
                for c in d["fatal_accident_cross_check"]]
        out += ["", "Extraction warnings: " + ("none" if not d["extraction_warnings"] else ""), ""]
        out += [f"- {w}" for w in d["extraction_warnings"]]
    if report["unreadable_documents"]:
        out += ["## Unreadable documents", ""]
        out += [f"- `{f['file']}`: {f['error']}" for f in report["unreadable_documents"]]
    out += ["", "## Mines and categories", "",
            "| Category | Mines | With water-related records | Identified by name only | Incidents | Years |",
            "|---|---|---|---|---|---|"]
    for c in report["by_category"].values():
        out.append(f"| {c['label']} | {c['mines']} | {c['mines_with_water_related_records']} | "
                   f"{c['mines_identified_by_name_only']} | {c['incidents']} | "
                   f"{c['year_range'][0]}-{c['year_range'][1]} |")
    out += ["", "## Water-related records (with source pages)", "",
            "| Mine | Category | Location | Date (as printed) | Cause in source | Basis | Killed | Source pages |",
            "|---|---|---|---|---|---|---|---|"]
    for w in sorted(report["water_related_records"], key=lambda x: (x["mine_category"], x["year"] or 0)):
        place = ", ".join(x for x in (w["district"], w["state"]) if x) or "not stated"
        pages = "; ".join(f"{s['source_filename']} p. {', '.join(map(str, s['pages']))}" for s in w["source_pages"])
        out.append(f"| {w['mine_name']} | {w['mine_category']} | {place} | {w['date_text']} | "
                   f"{w['cause_in_source']} | {w['water_relevance']} | {w['persons_killed']} | {pages} |")
    out += ["", "Basis legend: " + "; ".join(f"`{k}` = {v}" for k, v in WATER_LABELS.items() if k != "none"), ""]
    out += ["## Missing or incomplete fields", ""]
    out += [f"- {k.replace('_', ' ')}: {v}" for k, v in report["field_completeness"].items()]
    out += ["", "### Not present in the source volumes at all", ""]
    out += [f"- {x}" for x in report["fields_absent_from_source"]]
    out += ["", "## Ambiguous or flagged records", ""]
    if not report["ambiguous_or_flagged_records"]:
        out.append("None.")
    for a in report["ambiguous_or_flagged_records"]:
        out.append(f"- `{a['source_id']}` ({a['source_filename']} p. {', '.join(map(str, a['pages']))}, "
                   f"{SECTION_LABELS.get(a['section'], a['section'])}): " + " ".join(a["warnings"]))
    out += ["", "## Record merges (same accident reported in more than one section)", ""]
    out += [f"- `{m['merged_source']}` -> `{m['kept']}` ({m['basis']})" for m in report["record_merges"]] or ["None."]
    out += ["", "## Spelling aliases applied (curated file)", ""]
    out += [f"- {a['variant']} -> {a['canonical_key']} ({a['incident_id']})" for a in report["aliases_applied"]] or ["None."]
    out += ["", "## Likely duplicate mines NOT merged (for human review)", ""]
    out += [f"- {d['mine_a']['name']} ({d['mine_a']['district'] or '-'}, {d['mine_a']['state'] or '-'}) vs "
            f"{d['mine_b']['name']} ({d['mine_b']['district'] or '-'}, {d['mine_b']['state'] or '-'}): {d['reason']}"
            for d in report["likely_duplicates_not_merged"]] or ["None."]
    out += ["", "## Schema validation", ""]
    out += [f"- {w}" for w in report["schema_validation_warnings"]] or ["No integrity problems found."]
    out += ["", "## Vector index", "", f"- {json.dumps(report['vector_index'])}", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- pipeline
def run_ingestion(settings: Settings | None = None, *, rebuild: bool = False,
                  skip_vectors: bool = False) -> dict:
    """Run the pipeline and return the audit report. Raises IngestionError on a blocker."""
    settings = settings or get_settings()
    pdfs = discover_pdfs(settings)
    searched = ", ".join(str(p) for p in settings.pdf_dirs)
    if not pdfs:
        raise IngestionError(
            f"No PDF files were found in: {searched}. Place the two SANKET volumes (coal and "
            "non-coal) in one of these folders or set SANKET_PDF_DIRS in .env."
        )
    db = Database(settings.db_path)
    fingerprint = _fingerprint(pdfs, settings.alias_file)
    if not rebuild and db.exists() and settings.report_json.is_file():
        if db.meta().get("fingerprint") == fingerprint:
            log.info("Inputs unchanged; database is up to date. Use --rebuild to force a rebuild.")
            report = json.loads(settings.report_json.read_text(encoding="utf-8"))
            if not skip_vectors:
                report["vector_index"] = _sync_vectors(settings, db.all_chunks())
                settings.report_json.write_text(dump_json(report), encoding="utf-8")
                settings.report_md.write_text(render_report_markdown(report), encoding="utf-8")
            report["skipped"] = True
            return report

    results: list[ExtractionResult] = []
    failures: list[dict] = []
    for path in pdfs:
        try:
            results.append(extract_document(path))
        except PdfExtractionError as exc:
            log.error("%s", exc)
            failures.append({"file": path.name, "error": str(exc)})

    usable = [r for r in results if r.document.category]
    seen: dict[str, str] = {}
    for r in list(usable):
        cat = r.document.category
        if cat in seen:
            failures.append({"file": r.document.filename,
                             "error": f"Second volume for category '{cat}' (already have {seen[cat]}); ignored."})
            usable.remove(r)
        else:
            seen[cat] = r.document.filename
    for r in results:
        if not r.document.category:
            failures.append({"file": r.document.filename, "error": "; ".join(r.warnings)})
    missing = [CATEGORY_LABELS[c] for c in (CATEGORY_COAL, CATEGORY_NON_COAL) if c not in seen]
    if not usable:
        raise IngestionError(
            "None of the PDFs found could be processed as a SANKET volume: "
            + "; ".join(f"{f['file']}: {f['error']}" for f in failures)
        )
    if missing:
        log.warning("No readable volume found for: %s. Those categories will be empty.", ", ".join(missing))

    ds = normalize(usable, load_aliases(settings.alias_file))
    if not ds.incidents:
        raise IngestionError("The PDFs were readable but no accident records could be parsed from them.")
    chunks = build_chunks(ds)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    db.write(
        mines=ds.mines, incidents=ds.incidents, findings=ds.findings, sources=ds.sources, chunks=chunks,
        documents=[{"filename": r.document.filename, "sha256": r.document.sha256,
                    "page_count": r.document.page_count, "category": r.document.category,
                    "title": r.document.title,
                    "records": len(r.fatal_records) + len(r.major_rows) + len(r.court_rows)} for r in usable],
        meta={"fingerprint": fingerprint, "pipeline_version": PIPELINE_VERSION,
              "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "missing_categories": json.dumps(missing)},
    )
    vector_status = ({"status": "skipped", "reason": "--skip-vectors"} if skip_vectors
                     else _sync_vectors(settings, chunks, force=True))
    report = build_report(usable, failures, ds, vector_status)
    report["missing_categories"] = missing
    settings.report_json.write_text(dump_json(report), encoding="utf-8")
    settings.report_md.write_text(render_report_markdown(report), encoding="utf-8")
    log.info("Database written to %s", settings.db_path)
    log.info("Audit written to %s and %s", settings.report_json, settings.report_md)
    return report


def _sync_vectors(settings: Settings, chunks: list[dict], *, force: bool = False) -> dict:
    """Bring the vector index in line with the chunks. ``force`` re-embeds everything."""
    store = VectorStore(settings.vector_dir, settings.embedding_backend)
    if not store.enabled:
        return {"status": "disabled", "reason": "SANKET_EMBEDDING_BACKEND=none"}
    try:
        if not force and store.count() == len(chunks) and chunks:
            existing = store._open().get(include=[])["ids"]
            if set(existing) == {c["chunk_id"] for c in chunks}:
                return {"status": "up_to_date", "backend": settings.embedding_backend, "total": len(chunks)}
        log.info("Embedding %d chunks locally (first run downloads the MiniLM model once)...", len(chunks))
        return {"status": "ok", "backend": settings.embedding_backend, **store.sync(chunks)}
    except VectorStoreUnavailable as exc:
        log.warning("%s - continuing with keyword retrieval only.", exc)
        return {"status": "unavailable", "reason": str(exc)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract, normalise, validate and index the SANKET PDFs.")
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if the inputs are unchanged")
    parser.add_argument("--skip-vectors", action="store_true", help="do not build the semantic index")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        report = run_ingestion(rebuild=args.rebuild, skip_vectors=args.skip_vectors)
    except IngestionError as exc:
        log.error("BLOCKED: %s", exc)
        return 2
    t = report["totals"]
    print(f"\nMines: {t['mines']} | incidents: {t['incidents']} (water-related: {t['water_related_incidents']}) "
          f"| findings: {t['preventive_findings']} | sources: {t['source_references']}")
    for cat in report["by_category"].values():
        print(f"  {cat['label']}: {cat['mines']} mines, {cat['mines_with_water_related_records']} with water-related records")
    print(f"Vector index: {report['vector_index'].get('status')}")
    print(f"Audit: {get_settings().report_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
