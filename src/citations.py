"""Citation formatting, evidence serialisation for the model, and citation verification."""
from __future__ import annotations

import re
from pathlib import Path

from .schemas import CATEGORY_LABELS, WATER_LABELS, EvidenceBundle, Incident, SourceReference


def format_citation(source: SourceReference) -> str:
    return source.citation()


def citations_for(incident: Incident, sources: dict[str, SourceReference]) -> list[str]:
    return [sources[s].citation() for s in incident.source_ids if s in sources]


def citation_map(bundle: EvidenceBundle) -> dict[str, str]:
    return {sid: src.citation() for sid, src in bundle.sources.items()}


def redact_person(entry: str) -> str:
    """'1. P.Nonia, Trammer, Male, 57 Years' -> 'Trammer, Male, 57 Years' (drops the name)."""
    text = re.sub(r"^\s*\d+\.\s*", "", entry)
    parts = [p.strip() for p in text.split(",")]
    return ", ".join(parts[1:]) if len(parts) > 1 else "details not parsed"


def _safe(text: str | None) -> str:
    """Neutralise angle brackets so document text cannot close or open evidence tags."""
    return (text or "").replace("<", "‹").replace(">", "›")


def render_evidence(bundle: EvidenceBundle, *, redact_names: bool = True) -> str:
    """Serialise the evidence bundle for the model, with source ids on every record."""
    m = bundle.mine
    place = ", ".join(x for x in (m.district, m.state) if x) or "NOT DOCUMENTED IN SOURCE"
    lines = [
        "<evidence>",
        "<mine>",
        f"mine_id: {m.mine_id}",
        f"name: {_safe(m.mine_name)}",
        f"category: {CATEGORY_LABELS[m.mine_category]}",
        f"location (district, state): {_safe(place)}",
        f"owner/operator as printed: {_safe(m.operator) or 'NOT DOCUMENTED IN SOURCE'}",
        f"identity basis: {m.identity_basis}",
        f"water-related records: {len(bundle.incidents)}; other fatal-accident records: {len(bundle.other_incidents)}",
        "geology / hydrogeology: NOT AVAILABLE",
        "</mine>",
    ]
    findings_by_incident: dict[str, list] = {}
    for f in bundle.findings:
        findings_by_incident.setdefault(f.incident_id, []).append(f)
    lines.append("<water_related_incidents>")
    if not bundle.incidents:
        lines.append("NONE FOUND IN THE PROCESSED DOCUMENTS FOR THIS MINE.")
    for inc in bundle.incidents:
        cites = "; ".join(bundle.sources[s].citation() for s in inc.source_ids if s in bundle.sources)
        lines += [
            f'<incident incident_id="{inc.incident_id}" source_ids="{", ".join(inc.source_ids)}">',
            f"citation: {cites}",
            f"date as printed: {inc.date_text} (year {inc.incident_year}; date quality: {inc.date_quality})",
            f"time as printed: {inc.time_text or 'not stated'}",
            f"cause as classified in source: {_safe(inc.cause_label) or 'not stated'}",
            f"water relevance: {WATER_LABELS[inc.water_relevance]}. Basis: {_safe(inc.water_basis)}",
            f"persons killed: {inc.persons_killed_count if inc.persons_killed_count is not None else 'not stated'}"
            f"; seriously injured: {inc.seriously_injured_count if inc.seriously_injured_count is not None else 'not stated'}",
        ]
        if inc.persons_killed:
            people = [redact_person(p) if redact_names else p for p in inc.persons_killed]
            lines.append("casualties (occupation, sex, age): " + " | ".join(_safe(p) for p in people))
        if inc.court_of_inquiry:
            lines.append(f"court of inquiry: {_safe(inc.court_of_inquiry)}")
        lines.append("narrative (verbatim from source): " + (_safe(inc.description) or "NONE - table entry only"))
        items = findings_by_incident.get(inc.incident_id, [])
        if items:
            lines.append("DGMS findings - precautions NOT in place at the time (verbatim, each begins 'Had ...'):")
            lines += [f"  [{f.measure_id}] Had {_safe(f.measure_text)}" for f in items]
        else:
            lines.append("DGMS findings: NONE DOCUMENTED")
        lines.append("</incident>")
    lines.append("</water_related_incidents>")

    lines.append("<other_fatal_accidents_at_this_mine note=\"not water-related; background only\">")
    lines += [f"- {i.date_text}: {_safe(i.cause_label) or 'cause not stated'} (source_id {i.source_ids[0]})"
              for i in bundle.other_incidents[:25]] or ["NONE"]
    lines.append("</other_fatal_accidents_at_this_mine>")

    lines.append("<context>")
    lines += [f"- [{c.kind}] {_safe(c.title)}: {_safe(c.text)} (provenance: {_safe(c.provenance)}"
              + (f"; retrieved {c.retrieved_on}" if c.retrieved_on else "") + ")"
              for c in bundle.context.items] or ["NONE"]
    lines += [f"- LIMITATION: {x}" for x in bundle.context.limitations]
    lines.append("</context>")
    lines.append("<data_gaps>")
    lines += [f"- {g}" for g in bundle.data_gaps]
    lines += ["</data_gaps>", "</evidence>"]
    return "\n".join(lines)


def evidence_corpus(bundle: EvidenceBundle) -> str:
    """All text a factual claim may legitimately draw on (used by the safety validator)."""
    parts = [bundle.mine.mine_name, bundle.mine.state or "", bundle.mine.district or "",
             bundle.mine.operator or "", *bundle.mine.name_variants]
    parts += [s.source_text for s in bundle.sources.values()]
    parts += [s.citation() for s in bundle.sources.values()]
    for inc in [*bundle.incidents, *bundle.other_incidents]:
        parts += [inc.date_text, str(inc.incident_year or ""), inc.incident_date or "",
                  inc.description or "", inc.findings_text or "", inc.court_of_inquiry or "",
                  str(inc.persons_killed_count or ""), str(inc.seriously_injured_count or "")]
    parts += [c.text for c in bundle.context.items]
    return "\n".join(parts)


# --------------------------------------------------------------------------- verification
def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def verify_source_against_pdf(source: SourceReference, pdf_path: Path) -> dict:
    """Re-open the PDF and check that the stored excerpt really is on the cited pages.

    Returns the share of the excerpt's text lines found on those pages (1.0 = fully verified).
    """
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        page_text = _squash(" ".join(
            doc[p - 1].get_text("text") for p in source.pages if 0 < p <= doc.page_count
        ))
    fragments = [f for f in re.split(r"[\n|]", source.source_text) if len(_squash(f)) >= 4]
    found = sum(1 for f in fragments if _squash(f) in page_text)
    return {"source_id": source.source_id, "fragments": len(fragments), "found": found,
            "ratio": (found / len(fragments)) if fragments else 0.0}
