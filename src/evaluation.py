"""Evaluation suite over the real extracted dataset.

    python -m src.evaluation

Measures, for every mine that has water-related records:

* mine-identity accuracy  - retrieval returns only records whose mine_id is the requested one;
* citation correctness    - each cited excerpt is found on the cited pages of the source PDF;
* retrieval relevance     - a passage taken from an incident ranks that incident first
                            (keyword-only and hybrid keyword+semantic);
* unsupported-claim rate  - share of plan statements the safety validator had to remove.

Results are printed and written to data/processed/evaluation_report.json. The plan metric
always uses the offline rule engine so results are repeatable and no model quota is spent.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone

from .agent import SanketAgent
from .citations import verify_source_against_pdf
from .config import configure_logging, get_settings
from .database import DatabaseNotReady
from .ingestion import discover_pdfs
from .retrievers import Retriever


def _probe(text: str) -> str:
    """A mid-narrative passage (about 25 words) used as a retrieval query."""
    words = re.findall(r"[A-Za-z]{3,}", text)
    start = max(0, len(words) // 3)
    return " ".join(words[start : start + 25])


def run_evaluation() -> dict:
    settings = get_settings()
    retriever = Retriever.from_settings(settings)
    db = retriever.db
    agent = SanketAgent(retriever, settings)
    pdfs = {p.name: p for p in discover_pdfs(settings)}
    mines = db.list_mines(water_only=True)

    identity_total = identity_ok = 0
    cite_total = cite_ok = 0
    rel_total = rel_kw = rel_hybrid = 0
    steps = backed = removed = issues = plans_passed = 0
    unverified: list[str] = []
    keyword_only = Retriever(db, None, settings)

    for mine in mines:
        incidents = retriever.get_inundation_history(mine.mine_id, include_non_water=True)
        findings = retriever.get_preventive_findings(mine.mine_id)
        for record in [*incidents, *findings]:
            identity_total += 1
            identity_ok += record.mine_id == mine.mine_id
        for inc in (i for i in incidents if i.water_relevance != "none"):
            for src in db.get_sources(inc.source_ids).values():
                cite_total += 1
                pdf = pdfs.get(src.source_filename)
                ratio = verify_source_against_pdf(src, pdf)["ratio"] if pdf else 0.0
                if ratio >= 0.95:
                    cite_ok += 1
                else:
                    unverified.append(f"{src.source_id} ({ratio:.2f})")
            if inc.description and len(incidents) > 1:
                query = _probe(inc.description)
                rel_total += 1
                kw = keyword_only.hybrid_search(mine.mine_id, query, 3)
                hy = retriever.hybrid_search(mine.mine_id, query, 3)
                rel_kw += bool(kw) and kw[0].incident_id == inc.incident_id
                rel_hybrid += bool(hy) and hy[0].incident_id == inc.incident_id
        result = agent.run(mine.mine_id, use_llm=False)  # deterministic, free, repeatable
        v = result.plan.validation
        steps += v.steps_total
        backed += v.steps_evidence_backed
        removed += v.claims_removed
        issues += len(v.issues)
        plans_passed += v.passed
        for stage in result.plan.stages:
            for s in stage.steps:
                identity_total += len(s.evidence_source_ids)
                identity_ok += sum(sid in result.bundle.sources for sid in s.evidence_source_ids)

    def pct(a: int, b: int) -> float | None:
        return round(100.0 * a / b, 2) if b else None

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mines_evaluated": len(mines),
        "plan_generator": "offline rule engine",
        "semantic_index": "available" if retriever.vector_error is None and retriever.vectors.count() else "unavailable",
        "mine_identity_accuracy": {"checked": identity_total, "correct": identity_ok, "percent": pct(identity_ok, identity_total)},
        "citation_correctness": {"checked": cite_total, "verified_on_cited_pages": cite_ok,
                                 "percent": pct(cite_ok, cite_total), "not_verified": unverified},
        "retrieval_relevance_top1": {
            "queries": rel_total,
            "note": "Only mines with more than one record are queried; a single-record mine is trivially correct.",
            "keyword_only_percent": pct(rel_kw, rel_total), "hybrid_percent": pct(rel_hybrid, rel_total)},
        "plans": {"generated": len(mines), "passed_validation": plans_passed, "steps": steps,
                  "evidence_backed_steps": backed, "validator_issues": issues,
                  "unsupported_statements_removed": removed,
                  "unsupported_claim_rate_percent": pct(removed, steps + removed)},
    }
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    (settings.processed_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    configure_logging("WARNING")
    try:
        report = run_evaluation()
    except DatabaseNotReady as exc:
        print(exc)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
