"""Controlled, tool-using workflow that produces the emergency decision-support plan.

The model is never asked an open question. The orchestrator runs a fixed sequence of tools,
assembles an evidence bundle for exactly one mine_id, lets the generator (LLM or the offline
rule engine) write only the plan draft, and then validates that draft against the bundle.
Mine identity, incident lists, findings and citations in the final output always come from
the structured database, not from generated text.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .citations import citation_map, render_evidence
from .config import Settings, get_settings
from .llm import LLMUnavailable, generate_plan_draft, is_llm_enabled
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retrievers import Retriever
from .safety_validation import validate_plan
from .schemas import (
    STAGE_TITLES,
    DocumentedFinding,
    EmergencyPlan,
    EvidenceBundle,
    Incident,
    IncidentEvidence,
    Observation,
    PlanDraft,
    PlanStage,
    PlanStep,
    PreventiveFinding,
)

log = logging.getLogger(__name__)

DISCLAIMER = (
    "Decision-support aid only. This plan is generated from historical DGMS SANKET records and "
    "general safety practice. It is not an official order and does not replace the mine's approved "
    "emergency plan, statutory requirements, real-time monitoring, professional engineering "
    "judgment or the directions of competent authorities. Every item must be verified and "
    "authorized by qualified personnel before action is taken."
)


# --------------------------------------------------------------------------- tool schemas
class MineSearchArgs(BaseModel):
    query: str = Field(default="", description="Mine name or location text.")
    category: str | None = Field(default=None, description="'coal' or 'metal_non_coal'.")
    state: str | None = None
    district: str | None = None


class MineIdArgs(BaseModel):
    mine_id: str = Field(description="Stable mine identifier.")


class HistoryArgs(BaseModel):
    mine_id: str
    year_from: int | None = None
    year_to: int | None = None
    incident_types: list[str] | None = Field(
        default=None, description="Any of 'irruption', 'drowning', 'water_keyword'.")


class FindingsArgs(BaseModel):
    mine_id: str
    incident_ids: list[str] | None = None


@dataclass
class AgentResult:
    plan: EmergencyPlan
    bundle: EvidenceBundle
    trace: list[dict] = field(default_factory=list)
    llm_used: bool = False
    llm_error: str | None = None


class MineNotFound(LookupError):
    pass


# --------------------------------------------------------------------------- offline engine
def _quote(text: str, limit: int = 340) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


# finding category -> (stage, action, role, priority, verify, approvals)
_CATEGORY_STEPS: dict[str | None, tuple] = {
    "Water barrier / dam / stopping integrity": (
        3, "Have a qualified engineer assess every dam, stopping, retaining wall, embankment or parting that "
           "holds back water or sludge, against the head of water it may have to withstand.",
        "Competent mining / geotechnical engineer, reporting to the mine manager", "High",
        ["Design basis and present condition of each water-retaining structure",
         "Present water level or head behind each structure"],
        ["Engineering assessment by qualified personnel", "Mine manager's authorization for any remedial work"]),
    "Withdrawal of persons / danger level": (
        2, "Confirm that withdrawal and danger levels for surface water and waterlogged workings are defined, "
           "displayed and enforced before anyone is sent, or allowed to remain, below ground.",
        "Mine manager and safety officer", "Immediate",
        ["Current water level at the reference points against the fixed withdrawal level",
         "Record of persons below ground"],
        ["Mine manager's decision on withdrawal and re-entry"]),
    "Pumping / dewatering arrangement": (
        4, "Inspect the pumping and dewatering arrangements, including the siting of pump stations and the "
           "safety of persons working at sumps, before relying on them in an emergency.",
        "Engineer in charge of pumping, with the safety officer", "High",
        ["Pump availability, capacity actually delivered and standby power",
         "Safe access to and position of pump stations"],
        ["Qualified assessment before any change to pumping from behind a barrier"]),
    "Fencing / guarding of water bodies": (
        4, "Inspect fencing, guarding and edge protection at sumps, ponds, tanks and other water bodies.",
        "Safety officer", "Medium",
        ["Condition and continuity of fencing and edge protection"], ["Mine manager's approval of corrective work"]),
    "Life-saving equipment": (
        4, "Verify that lifelines, life jackets or equivalent protection are available and used wherever "
           "persons work at or over water.",
        "Safety officer", "Medium",
        ["Availability and condition of life-saving equipment at each water body"], []),
    "Risk assessment / SOP / code of practice": (
        3, "Check that a current risk assessment and a written safe operating procedure cover the activity "
           "described in the historical record, and that they are being followed.",
        "Mine manager and safety officer", "High",
        ["Date and scope of the latest risk assessment", "Evidence that the procedure is followed on site"],
        ["Manager's approval of the procedure"]),
    "Survey / plans / old workings": (
        3, "Have the surveyor confirm, from up-to-date plans, the position of old, disused or waterlogged "
           "workings and water bodies relative to current workings.",
        "Mine surveyor, reporting to the mine manager", "High",
        ["Accuracy and date of the mine plans", "Position of waterlogged workings and surface water bodies"],
        ["Review by qualified mining personnel"]),
    "Supervision / competent person": (
        5, "Confirm that the supervisory and competent-person appointments for the affected operations are "
           "filled and that supervision is in place on every shift.",
        "Mine manager", "Medium", ["Current statutory appointments and shift deployment"], []),
    "Training / awareness": (
        6, "Include the circumstances of the documented accident in refresher training and mock rehearsals.",
        "Safety officer / training officer", "Routine", ["Training and mock-rehearsal records"], []),
    None: (
        3, "Review the DGMS finding quoted below with qualified personnel and determine whether the "
           "precaution it describes is in place today.",
        "Mine manager and safety officer", "Medium", ["Present status of the precaution described"], []),
}


def build_offline_draft(bundle: EvidenceBundle) -> PlanDraft:
    """Deterministic plan: general-practice steps plus steps tied to this mine's records."""
    mine, src = bundle.mine, bundle.sources
    steps: dict[int, list[dict]] = {n: [] for n in range(1, 7)}

    def cite(inc: Incident) -> str:
        return "; ".join(src[s].citation() for s in inc.source_ids if s in src)

    def add(stage: int, action: str, relevance: str, role: str, priority: str, *, evidence: str | None = None,
            ids: list[str] | None = None, verify: list[str] | None = None, approvals: list[str] | None = None,
            first: bool = False) -> None:
        item = dict(action=action, relevance_to_this_mine=relevance, historical_evidence=evidence,
                    evidence_source_ids=ids or [], basis="documented_evidence" if evidence else "general_practice",
                    responsible_role=role, priority=priority, verify_before_acting=verify or [],
                    dependencies_and_approvals=approvals or [])
        steps[stage].insert(0, item) if first else steps[stage].append(item)

    generic = "General safety practice; not derived from this mine's records."
    narrated = [i for i in bundle.incidents if i.description]
    table_only = [i for i in bundle.incidents if not i.description]

    # ---- stage 1
    for inc in narrated[:3]:
        add(1, f"Treat the circumstances described in the record of {inc.date_text} as warning signs to check "
               "first when water ingress or rising water is reported at this mine.",
            f"This mine has a documented water-related fatal accident ({inc.cause_label or 'cause not stated'}).",
            "Mine manager and safety officer", "Immediate",
            evidence=f"DGMS record of {inc.date_text} ({cite(inc)}): \"{_quote(inc.description)}\"",
            ids=inc.source_ids,
            verify=["Whether the conditions described in the record exist at present",
                    "Current water levels at the places named in the record"])
    add(1, "Verify any report of water ingress at the reported place through the mine's monitoring arrangements "
           "and inspection by a competent person, without delaying withdrawal where danger is apparent.",
        generic, "Shift in-charge / competent person", "Immediate",
        verify=["Who reported, where, and when", "Readings from available water-level and pumping monitoring"])
    add(1, "Establish the facts needed for decisions: location and time of ingress, whether inflow is rising, "
           "persons below ground and their locations, state of escape routes, power and pumping status.",
        generic, "Mine manager", "Immediate",
        verify=["Attendance and deployment records", "Condition of travelling roadways and escape routes"])

    # ---- stage 2
    add(2, "Withdraw persons from threatened workings in accordance with the mine's approved emergency plan and "
           "account for every person.",
        generic, "Mine manager", "Immediate",
        verify=["Head count against the deployment record"], approvals=["Mine manager's withdrawal decision"])
    add(2, "Notify the owner/agent, the DGMS regional office and the district administration as required by the "
           "applicable statute and the emergency plan, and activate the rescue arrangements.",
        generic, "Mine manager", "Immediate",
        verify=["Current contact details in the emergency plan"],
        approvals=["Statutory notification requirements"])
    add(2, "Restrict entry to affected areas. Allow re-entry only on the manager's authorization after a "
           "competent assessment.",
        generic, "Mine manager and safety officer", "High", approvals=["Manager's written authorization for re-entry"])

    # ---- stage 3 (table-only history)
    for inc in table_only[:4]:
        killed = f"{inc.persons_killed_count} person(s) killed" if inc.persons_killed_count is not None else "casualties not stated"
        add(3, f"Obtain the inquiry report or DGMS file for the accident of {inc.date_text} before assuming "
               "anything about the source or path of water in that event.",
            "The source lists a water-related major accident under this mine name but gives no mechanism, "
            "affected workings or findings.",
            "Mine manager, with district administration support for record retrieval", "Medium",
            evidence=f"Table entry ({cite(inc)}): date {inc.date_text}; cause as printed "
                     f"'{inc.cause_label}'; {killed}."
                     + (f" {inc.court_of_inquiry}" if inc.court_of_inquiry else ""),
            ids=inc.source_ids,
            verify=["That the table entry refers to this mine (the table prints the name only)"])
    add(3, "Have qualified mining, geotechnical and survey personnel assess the source of water, the head of "
           "water and the stability of barriers, using current mine plans.",
        generic, "Mine manager with qualified technical personnel", "High",
        verify=["Up-to-date plans and sections", "Extent of waterlogged or disused workings"],
        approvals=["Technical assessment before any intervention"])

    # ---- category-driven steps from DGMS findings
    by_cat: dict[str | None, list[PreventiveFinding]] = {}
    for f in bundle.findings:
        by_cat.setdefault(f.measure_category, []).append(f)
    incident_by_id = {i.incident_id: i for i in bundle.incidents}
    for category, items in by_cat.items():
        stage, action, role, priority, verify, approvals = _CATEGORY_STEPS.get(category, _CATEGORY_STEPS[None])
        quotes = "; ".join(
            f"accident of {incident_by_id[f.incident_id].date_text} "
            f"({src[f.source_ids[0]].citation()}): \"Had {_quote(f.measure_text, 260)}\""
            for f in items[:2]
        )
        add(stage, action,
            "DGMS found that this precaution was not in place when a water-related fatal accident occurred at "
            "this mine. The record does not say whether it was implemented afterwards.",
            role, priority, evidence=f"DGMS finding, {quotes}",
            ids=list(dict.fromkeys(s for f in items[:2] for s in f.source_ids)),
            verify=verify + ["Whether this precaution is in place today (not documented)"], approvals=approvals,
            first=True)

    # ---- stage 4
    add(4, "Inspect and verify pumping, drainage and power arrangements before relying on them. Treat any proposal "
           "to pump out, dam, seal, drill or divert water as an engineering recommendation that requires "
           "assessment and approval by qualified personnel.",
        generic, "Mine manager with qualified engineers", "High",
        verify=["Installed and standby pumping capacity", "Power supply security"],
        approvals=["Qualified technical assessment", "Approval of the competent authority where required"])

    # ---- stage 5
    if bundle.incidents:
        listing = "; ".join(
            f"{i.date_text} ({i.cause_label or 'cause not stated'}, "
            f"{i.persons_killed_count if i.persons_killed_count is not None else 'unknown number'} killed)"
            for i in bundle.incidents[:8])
        add(5, "Brief the district administration and responding agencies on this mine's documented water-related "
               "history, stating clearly which details are unknown.",
            "Decision makers new to the area need the documented history and its limits.",
            "Mine manager / safety officer", "High",
            evidence=f"The processed SANKET volumes contain {len(bundle.incidents)} water-related record(s) "
                     f"for this mine: {listing}.",
            ids=list(dict.fromkeys(s for i in bundle.incidents[:8] for s in i.source_ids)))
    add(5, "Maintain one coordination channel between the mine manager, safety personnel, district administration, "
           "DGMS and rescue services, and communicate: location, persons affected, actions taken, resources "
           "needed and decisions awaiting approval.",
        generic, "District administration with the mine manager", "High",
        approvals=["Decisions on rescue entry, large-scale pumping and public warnings rest with authorized officials"])

    # ---- stage 6
    if bundle.findings:
        add(6, "Review each precaution that DGMS found was not in place in this mine's past water-related "
               "accidents and record, with evidence, whether it is in place today.",
            "The findings identify what would have averted past accidents here. Their present status is unknown.",
            "Mine manager and safety officer", "High",
            evidence=f"{len(bundle.findings)} DGMS finding(s) are documented for this mine's water-related "
                     "accidents. The records do not say whether any were later implemented.",
            ids=list(dict.fromkeys(s for f in bundle.findings for s in f.source_ids)),
            verify=["Present status of each precaution"])
    add(6, "Inspect affected workings and installations, document the incident fully, and carry out a root-cause "
           "investigation.",
        generic, "Mine manager; investigating authority as applicable", "High",
        approvals=["Statutory inquiry requirements"])
    add(6, "Resume operations only in a controlled manner, after corrective actions are completed and the "
           "applicable approvals are obtained.",
        generic, "Mine manager; competent regulatory authority", "High",
        approvals=["Clearance from the competent authority where required"])

    n = 0
    stages = []
    for number, title in enumerate(STAGE_TITLES, 1):
        built = []
        for item in steps[number]:
            n += 1
            built.append(PlanStep(step_number=n, **item))
        stages.append(PlanStage(stage_number=number, title=title, steps=built))

    observations: list[Observation] = []
    for inc in bundle.incidents[:10]:
        killed = (f"{inc.persons_killed_count} person(s) killed" if inc.persons_killed_count is not None
                  else "casualties not stated")
        observations.append(Observation(
            text=f"{inc.date_text}: the source classifies the cause as '{inc.cause_label or 'not stated'}'; {killed}.",
            kind="documented_fact", evidence_source_ids=inc.source_ids))
    cats = sorted(c for c in by_cat if c)
    if cats:
        observations.append(Observation(
            text="DGMS findings for this mine's water-related accidents concern: " + "; ".join(cats) + ".",
            kind="documented_fact",
            evidence_source_ids=list(dict.fromkeys(s for f in bundle.findings for s in f.source_ids))))
    if len(bundle.incidents) > 1:
        observations.append(Observation(
            text="More than one water-related record exists for this mine, which suggests that water hazards "
                 "deserve specific attention in its emergency planning. This is an inference from the records, "
                 "not a forecast.",
            kind="reasonable_inference"))

    if bundle.incidents:
        listing = "; ".join(
            f"{i.date_text} - {i.cause_label or 'cause not stated'}"
            + (f", {i.persons_killed_count} killed" if i.persons_killed_count is not None else "")
            for i in bundle.incidents)
        summary = (f"The processed SANKET volumes contain {len(bundle.incidents)} water-related record(s) for "
                   f"{mine.mine_name}: {listing}.")
    else:
        summary = "No water-related record for this mine was found in the processed SANKET documents."

    assumptions = [
        "The mine has an approved emergency plan and standing orders, and these take precedence over this document.",
        "Steps marked 'general practice' are not derived from this mine's records.",
        "Records were classified as water-related from DGMS cause codes and, where marked, from keywords in the "
        "narrative; a keyword match may describe water only as a secondary circumstance.",
        "This plan was assembled by a rule engine without a language model; it does not interpret the narratives "
        "beyond quoting them.",
    ]
    approvals = [
        "Withdrawal, re-entry and resumption decisions: mine manager, under the approved emergency plan.",
        "Any pumping from behind a barrier, damming, sealing, drilling or diversion: assessment and approval by "
        "qualified mining and geotechnical engineers.",
        "Statutory notifications and inquiry: owner/agent/manager and DGMS as applicable.",
        "Deployment of district resources and public warnings: district administration.",
        "Verification of every historical item against present site conditions before it is relied on.",
    ]
    return PlanDraft(historical_incident_summary=summary, mine_specific_observations=observations,
                     stages=stages, assumptions_and_uncertainties=assumptions,
                     missing_information=[], required_human_verification_and_approvals=approvals)


# --------------------------------------------------------------------------- orchestrator
class SanketAgent:
    def __init__(self, retriever: Retriever | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.retriever = retriever or Retriever.from_settings(self.settings)
        self._all_mine_names: list[str] | None = None
        self.tools: dict[str, StructuredTool] = {t.name: t for t in self._build_tools()}

    def _build_tools(self) -> list[StructuredTool]:
        r = self.retriever
        return [
            StructuredTool.from_function(
                name="mine_directory_search", args_schema=MineSearchArgs,
                description="Find mines by name, category or documented location. Returns ids and match quality.",
                func=lambda query="", category=None, state=None, district=None: r.search_mines(
                    query, category, state=state, district=district)),
            StructuredTool.from_function(
                name="mine_record_retriever", args_schema=MineIdArgs,
                description="Canonical record, incident ids and source references for one mine_id.",
                func=lambda mine_id: r.get_mine_record(mine_id)),
            StructuredTool.from_function(
                name="inundation_history_retriever", args_schema=HistoryArgs,
                description="Water-related incidents for exactly one mine_id, optionally by year or type.",
                func=lambda mine_id, year_from=None, year_to=None, incident_types=None: r.get_inundation_history(
                    mine_id, year_from=year_from, year_to=year_to, incident_types=incident_types)),
            StructuredTool.from_function(
                name="preventive_measures_retriever", args_schema=FindingsArgs,
                description="Documented DGMS preventive findings for one mine_id and its incidents.",
                func=lambda mine_id, incident_ids=None: r.get_preventive_findings(mine_id, incident_ids)),
            StructuredTool.from_function(
                name="context_and_guidance_retriever", args_schema=MineIdArgs,
                description="Verified context with provenance; reports the limitation when none exists.",
                func=lambda mine_id: r.get_context(mine_id)),
            StructuredTool.from_function(
                name="emergency_plan_generator", args_schema=MineIdArgs,
                description="Generate the validated, evidence-grounded emergency decision-support plan.",
                func=lambda mine_id: self.run(mine_id).plan),
        ]

    def _other_mine_names(self) -> list[str]:
        if self._all_mine_names is None:
            self._all_mine_names = [m.mine_name for m in self.retriever.db.list_mines()]
        return self._all_mine_names

    def run(self, mine_id: str, *, use_llm: bool | None = None) -> AgentResult:
        trace: list[dict] = []

        def step(label: str, tool: str | None, fn):
            start = time.perf_counter()
            out = fn()
            trace.append({"step": label, "tool": tool, "ms": round((time.perf_counter() - start) * 1000, 1)})
            return out

        # 1-2 resolve the canonical mine
        record = step("Resolve mine and load canonical record", "mine_record_retriever",
                      lambda: self.tools["mine_record_retriever"].invoke({"mine_id": mine_id}))
        if record is None:
            raise MineNotFound(f"No mine with id '{mine_id}' exists in the processed records.")
        # 3-5 retrieve history, findings, context (each strictly by mine_id)
        incidents = step("Retrieve water-related incidents", "inundation_history_retriever",
                         lambda: self.tools["inundation_history_retriever"].invoke({"mine_id": mine_id}))
        findings = step("Retrieve documented preventive findings", "preventive_measures_retriever",
                        lambda: self.tools["preventive_measures_retriever"].invoke(
                            {"mine_id": mine_id, "incident_ids": [i.incident_id for i in incidents]}))
        step("Retrieve verified context", "context_and_guidance_retriever",
             lambda: self.tools["context_and_guidance_retriever"].invoke({"mine_id": mine_id}))
        # 6 evidence bundle
        bundle = step("Assemble evidence bundle with citations", None,
                      lambda: self.retriever.build_evidence_bundle(mine_id))
        assert {i.incident_id for i in bundle.incidents} == {i.incident_id for i in incidents}
        assert all(i.mine_id == mine_id for i in bundle.incidents + bundle.other_incidents)
        assert all(f.mine_id == mine_id for f in findings)

        # 7 generate
        want_llm = is_llm_enabled(self.settings) if use_llm is None else use_llm
        llm_used, llm_error, draft = False, None, None
        if want_llm:
            try:
                prompt = build_user_prompt(
                    render_evidence(bundle, redact_names=self.settings.redact_person_names),
                    bundle.has_water_evidence)
                draft = step(f"Generate plan draft ({self.settings.llm_provider})", "emergency_plan_generator",
                             lambda: generate_plan_draft(SYSTEM_PROMPT, prompt, self.settings))
                llm_used = True
            except LLMUnavailable as exc:
                llm_error = str(exc)
                log.warning("LLM unavailable, using offline rule engine: %s", exc)
        if draft is None:
            draft = step("Generate plan draft (offline rule engine)", "emergency_plan_generator",
                         lambda: build_offline_draft(bundle))
        # 8-9 validate structure and evidential support
        cleaned, report = step("Validate schema and check claims against evidence", None,
                               lambda: validate_plan(draft, bundle, self._other_mine_names()))
        if llm_used and len(cleaned.stages) != 6:
            # an unusable model answer must not reach the user: fall back to the rule engine
            llm_error = "The model's plan failed validation (missing stages); the offline plan is shown instead."
            llm_used = False
            cleaned, report = validate_plan(build_offline_draft(bundle), bundle, self._other_mine_names())
        # 10 final assembly from structured records
        plan = step("Assemble final plan", None,
                    lambda: self._assemble(bundle, cleaned, report, llm_used, llm_error))
        return AgentResult(plan=plan, bundle=bundle, trace=trace, llm_used=llm_used, llm_error=llm_error)

    def _assemble(self, bundle: EvidenceBundle, draft: PlanDraft, report, llm_used: bool,
                  llm_error: str | None) -> EmergencyPlan:
        src = bundle.sources
        incident_evidence = [
            IncidentEvidence(
                incident_id=i.incident_id, date_text=i.date_text, incident_year=i.incident_year,
                cause_label=i.cause_label, water_relevance=i.water_relevance, description=i.description,
                persons_killed_count=i.persons_killed_count,
                citations=[src[s].citation() for s in i.source_ids if s in src], source_ids=i.source_ids)
            for i in bundle.incidents
        ]
        findings = [
            DocumentedFinding(
                measure_id=f.measure_id, incident_id=f.incident_id, measure_text=f.measure_text,
                measure_category=f.measure_category, current_status=f.current_status,
                citations=[src[s].citation() for s in f.source_ids if s in src], source_ids=f.source_ids)
            for f in bundle.findings
        ]
        assumptions = list(draft.assumptions_and_uncertainties)
        if llm_error:
            assumptions.insert(0, f"Language model not used: {llm_error}")
        missing = list(dict.fromkeys([*bundle.data_gaps, *draft.missing_information]))
        generated_by = (f"{self.settings.llm_provider}:{self.settings.default_model()} with evidence validation"
                        if llm_used else "Offline rule engine (no language model) with evidence validation")
        return EmergencyPlan(
            plan_type="mine_specific" if bundle.has_water_evidence else "generic_framework",
            mine=bundle.mine, historical_incident_summary=draft.historical_incident_summary,
            incident_evidence=incident_evidence, documented_preventive_findings=findings,
            mine_specific_observations=draft.mine_specific_observations, stages=draft.stages,
            evidence_citations=citation_map(bundle), assumptions_and_uncertainties=assumptions,
            missing_information=missing,
            required_human_verification_and_approvals=draft.required_human_verification_and_approvals,
            generated_by=generated_by,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            validation=report, disclaimer=DISCLAIMER,
        )
