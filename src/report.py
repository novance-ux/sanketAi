"""Plain-Markdown export of a generated plan, for filing or printing."""
from __future__ import annotations

from .schemas import CATEGORY_LABELS, EmergencyPlan


def plan_to_markdown(plan: EmergencyPlan) -> str:
    m = plan.mine
    place = ", ".join(x for x in (m.district, m.state) if x) or "not documented in source"
    kind = ("Mine-specific plan" if plan.plan_type == "mine_specific"
            else "GENERIC SAFETY FRAMEWORK - no mine-specific water-related record was found")
    out = [
        f"# SANKET AI - Emergency Decision-Support Plan: {m.mine_name}", "",
        f"> {plan.disclaimer}", "",
        f"- Plan type: {kind}",
        f"- Category: {CATEGORY_LABELS[m.mine_category]}",
        f"- Location: {place}",
        f"- Owner/operator as printed: {m.operator or 'not documented in source'}",
        f"- Mine ID: {m.mine_id}",
        f"- Generated: {plan.generated_at} by {plan.generated_by}", "",
        "## Historical incident summary", "", plan.historical_incident_summary, "",
        "## Incident-by-incident evidence (documented)", "",
    ]
    for inc in plan.incident_evidence:
        out += [f"### {inc.date_text} - {inc.cause_label or 'cause not stated'}",
                f"- Persons killed: {inc.persons_killed_count if inc.persons_killed_count is not None else 'not stated'}",
                f"- Source: {'; '.join(inc.citations)}", "",
                inc.description or "_Table entry only; no narrative in the source._", ""]
    if not plan.incident_evidence:
        out += ["No matching record was found in the processed documents.", ""]
    out += ["## Documented preventive findings (DGMS inquiry; precautions NOT in place at the time)", ""]
    for f in plan.documented_preventive_findings:
        out.append(f"- Had {f.measure_text} [{'; '.join(f.citations)}] - present status: {f.current_status}")
    if not plan.documented_preventive_findings:
        out.append("None documented.")
    out += ["", "## Mine-specific observations", ""]
    for o in plan.mine_specific_observations:
        tag = "Documented fact" if o.kind == "documented_fact" else "Inference"
        cites = "; ".join(plan.evidence_citations.get(s, s) for s in o.evidence_source_ids)
        out.append(f"- [{tag}] {o.text}" + (f" ({cites})" if cites else ""))
    if not plan.mine_specific_observations:
        out.append("None.")
    out += ["", "## Proposed emergency response steps", ""]
    for stage in plan.stages:
        out += [f"### Stage {stage.stage_number}: {stage.title}", ""]
        for s in stage.steps:
            basis = "Documented evidence" if s.basis == "documented_evidence" else "General practice"
            out += [f"**Step {s.step_number} [{s.priority}; {basis}]** {s.action}",
                    f"- Why relevant here: {s.relevance_to_this_mine}"]
            if s.historical_evidence:
                cites = "; ".join(plan.evidence_citations.get(x, x) for x in s.evidence_source_ids)
                out.append(f"- Historical evidence: {s.historical_evidence} [{cites}]")
            out.append(f"- Responsible: {s.responsible_role}")
            if s.verify_before_acting:
                out.append("- Verify first: " + "; ".join(s.verify_before_acting))
            if s.dependencies_and_approvals:
                out.append("- Dependencies and approvals: " + "; ".join(s.dependencies_and_approvals))
            out.append("")
    for title, items in (
        ("Assumptions and uncertainties", plan.assumptions_and_uncertainties),
        ("Missing information", plan.missing_information),
        ("Required human verification and approvals", plan.required_human_verification_and_approvals),
    ):
        out += [f"## {title}", ""] + [f"- {x}" for x in items] + [""]
    out += ["## Evidence citations", ""] + [f"- `{k}`: {v}" for k, v in plan.evidence_citations.items()]
    v = plan.validation
    out += ["", "## Validation", "",
            f"- Checks passed: {'yes' if v.passed else 'NO'}; steps: {v.steps_total}; "
            f"evidence-backed steps: {v.steps_evidence_backed}; statements removed: {v.claims_removed}"]
    out += [f"- {i.severity.upper()} {i.code} at {i.location}: {i.message} ({i.action_taken})" for i in v.issues]
    return "\n".join(out) + "\n"
