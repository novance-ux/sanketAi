"""Pydantic models: the normalised Sanket record schema and the agent's structured output.

The record schema was derived from the two source volumes after inspection, not assumed
beforehand. Fields the volumes never state (region, geology, water level as a separate
field, measures actually installed, present-day status) are deliberately absent or nullable
and are never filled with defaults.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MineCategory = Literal["coal", "metal_non_coal"]
CATEGORY_LABELS: dict[str, str] = {"coal": "Coal Mines", "metal_non_coal": "Metal / Non-Coal Mines"}

# How strongly a record relates to water, and on what documented basis.
WaterRelevance = Literal["irruption", "drowning", "water_keyword", "none"]
WATER_LABELS: dict[str, str] = {
    "irruption": "Irruption of water / inundation (DGMS cause classification)",
    "drowning": "Drowning in water (DGMS cause classification)",
    "water_keyword": "Water mentioned in narrative (DGMS classified under another cause)",
    "none": "Not water-related",
}
SECTION_LABELS: dict[str, str] = {
    "fatal_accident_summary": "Summary of findings of fatal accidents",
    "major_accident_table": "Major accidents since 1901 (table)",
    "court_of_inquiry_table": "Courts of inquiry (table)",
}


class SourceReference(BaseModel):
    source_id: str
    source_filename: str
    page_start: int
    page_end: int
    pages: list[int]
    section: str
    record_type: str  # incident
    source_text: str  # verbatim text as extracted from the PDF
    extraction_method: str
    verification_status: Literal["parsed_clean", "needs_review"]
    warnings: list[str] = Field(default_factory=list)

    def citation(self) -> str:
        pages = f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
        return f"{self.source_filename}, {pages}"


class Mine(BaseModel):
    mine_id: str
    mine_name: str
    mine_category: MineCategory
    name_variants: list[str] = Field(default_factory=list)
    state: str | None = None
    district: str | None = None
    operator: str | None = None
    operator_variants: list[str] = Field(default_factory=list)
    # What the identity rests on. Historic tables give a mine name and nothing else.
    identity_basis: Literal["name_state_district", "name_only"]
    geology_summary: str | None = None  # never present in the Sanket volumes
    incident_count: int = 0
    water_incident_count: int = 0
    irruption_count: int = 0
    drowning_count: int = 0
    water_keyword_count: int = 0
    earliest_year: int | None = None
    latest_year: int | None = None
    possibly_related_mine_ids: list[str] = Field(default_factory=list)


class Incident(BaseModel):
    incident_id: str
    mine_id: str
    mine_name_in_source: str
    incident_date: str | None = None  # ISO date, null when the source date is ambiguous
    incident_year: int | None = None
    date_text: str  # exactly as printed
    date_quality: Literal["exact", "century_inferred", "day_month_ambiguous", "unparsed"]
    time_text: str | None = None
    record_sections: list[str]
    cause_code: str | None = None
    cause_label: str | None = None
    cause_group: str | None = None
    water_relevance: WaterRelevance = "none"
    water_basis: str | None = None
    description: str | None = None  # DGMS narrative; null for table-only records
    findings_text: str | None = None  # verbatim "Had ... could have been averted" block
    persons_killed_count: int | None = None
    seriously_injured_count: int | None = None
    persons_killed: list[str] = Field(default_factory=list)
    measurements_mentioned: list[str] = Field(default_factory=list)
    regulations_cited: list[str] = Field(default_factory=list)
    court_of_inquiry: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PreventiveFinding(BaseModel):
    """One precaution that the DGMS inquiry found was NOT in place.

    The volumes phrase these as "Had <precaution> been <taken> ... this accident could have
    been averted". They are documented findings about what would have prevented the
    accident. They are not evidence that the precaution was later implemented.
    """

    measure_id: str
    mine_id: str
    incident_id: str
    measure_text: str  # verbatim item
    measure_kind: Literal["dgms_finding_precaution_not_in_place"] = "dgms_finding_precaution_not_in_place"
    measure_category: str | None = None
    category_method: str | None = None
    regulations_cited: list[str] = Field(default_factory=list)
    implementation_date: str | None = None  # not documented in the source
    reported_outcome: str | None = None  # not documented in the source
    current_status: Literal["unknown"] = "unknown"
    extraction_quality: Literal["itemised", "unsplit_block"] = "itemised"
    source_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- tool I/O
class MineMatch(BaseModel):
    mine_id: str
    mine_name: str
    mine_category: MineCategory
    state: str | None
    district: str | None
    operator: str | None
    water_incident_count: int
    incident_count: int
    match_type: Literal["exact", "prefix", "substring", "location", "fuzzy", "listing"]
    score: float


class MineSearchResult(BaseModel):
    query: str
    category: MineCategory | None
    matches: list[MineMatch]
    ambiguous: bool
    message: str


class ContextItem(BaseModel):
    title: str
    text: str
    provenance: str
    retrieved_on: str | None
    kind: Literal["in_corpus", "external"]


class ContextResult(BaseModel):
    items: list[ContextItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    """Everything the plan generator is allowed to rely on for one mine."""

    mine: Mine
    incidents: list[Incident]  # water-related, chronological
    other_incidents: list[Incident] = Field(default_factory=list)  # non-water, for context only
    findings: list[PreventiveFinding]
    sources: dict[str, SourceReference]
    context: ContextResult = Field(default_factory=ContextResult)
    data_gaps: list[str] = Field(default_factory=list)

    @property
    def has_water_evidence(self) -> bool:
        return bool(self.incidents)


# --------------------------------------------------------------------------- plan output
Priority = Literal["Immediate", "High", "Medium", "Routine"]
StepBasis = Literal["documented_evidence", "general_practice"]

STAGE_TITLES: tuple[str, ...] = (
    "Initial Detection and Verification",
    "Immediate Safety Actions",
    "Containment and Risk Assessment",
    "Water Management and Engineering Response",
    "Administrative Coordination",
    "Recovery and Prevention",
)


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_number: int = Field(ge=1)
    action: str = Field(min_length=5, description="Recommended action, phrased as a recommendation.")
    relevance_to_this_mine: str = Field(description="Why this step matters for this mine.")
    historical_evidence: str | None = Field(
        default=None,
        description="What the cited record says. Null when the step is general practice.",
    )
    evidence_source_ids: list[str] = Field(
        default_factory=list, description="source_id values from the evidence bundle only."
    )
    basis: StepBasis
    responsible_role: str
    priority: Priority
    verify_before_acting: list[str] = Field(default_factory=list)
    dependencies_and_approvals: list[str] = Field(default_factory=list)


class PlanStage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    stage_number: int = Field(ge=1, le=6)
    title: str
    steps: list[PlanStep] = Field(min_length=1)


class Observation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    kind: Literal["documented_fact", "reasonable_inference"]
    evidence_source_ids: list[str] = Field(default_factory=list)


class PlanDraft(BaseModel):
    """The part of the plan written by the model (or by the offline rule engine)."""

    model_config = ConfigDict(extra="ignore")

    historical_incident_summary: str
    mine_specific_observations: list[Observation] = Field(default_factory=list)
    stages: list[PlanStage] = Field(min_length=6, max_length=6)
    assumptions_and_uncertainties: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    required_human_verification_and_approvals: list[str] = Field(default_factory=list)


class IncidentEvidence(BaseModel):
    incident_id: str
    date_text: str
    incident_year: int | None
    cause_label: str | None
    water_relevance: WaterRelevance
    description: str | None
    persons_killed_count: int | None
    citations: list[str]
    source_ids: list[str]


class DocumentedFinding(BaseModel):
    measure_id: str
    incident_id: str
    measure_text: str
    measure_category: str | None
    current_status: str
    citations: list[str]
    source_ids: list[str]


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    location: str
    message: str
    action_taken: str


class ValidationReport(BaseModel):
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    steps_total: int = 0
    steps_evidence_backed: int = 0
    claims_removed: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]


class EmergencyPlan(BaseModel):
    """Final structured decision-support output rendered by the UI."""

    plan_type: Literal["mine_specific", "generic_framework"]
    mine: Mine
    historical_incident_summary: str
    incident_evidence: list[IncidentEvidence]
    documented_preventive_findings: list[DocumentedFinding]
    mine_specific_observations: list[Observation]
    stages: list[PlanStage]
    evidence_citations: dict[str, str]  # source_id -> "file, p. N"
    assumptions_and_uncertainties: list[str]
    missing_information: list[str]
    required_human_verification_and_approvals: list[str]
    generated_by: str
    generated_at: str
    validation: ValidationReport
    disclaimer: str
