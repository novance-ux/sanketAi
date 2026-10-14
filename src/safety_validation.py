"""Post-generation safety checks on a draft plan (orchestration step 9).

The validator does not trust the generator - LLM or rule engine. It checks every draft
against the evidence bundle and repairs or removes anything it cannot support:

* citations must point at sources in this mine's bundle;
* a step marked 'documented_evidence' must actually cite something;
* numbers and dates in factual text must occur in the evidence;
* other mines' names must not appear;
* no probabilities, return periods, predictions or guarantees;
* no claim that a historical measure is in place today;
* no wording that reads as an official order.

Every intervention is recorded in the ValidationReport that the UI shows to the user.
"""
from __future__ import annotations

import re

from .citations import evidence_corpus
from .data_normalizer import norm_name
from .schemas import (
    STAGE_TITLES,
    EvidenceBundle,
    Observation,
    PlanDraft,
    PlanStage,
    PlanStep,
    ValidationIssue,
    ValidationReport,
)

_PREDICTIVE = re.compile(
    r"\b\d{1,3}(?:\.\d+)?\s?(?:%|per ?cent)\s*(?:chance|probability|likelihood|risk)"
    r"|\b(?:probability|likelihood|chance)\s+(?:of|is|=)\s*(?:about\s+|approximately\s+)?\d"
    r"|\breturn period\b|\b1[- ]in[- ]\d+[- ]years?\b|\b\d+[- ]year (?:flood|event|return)"
    r"|\brisk score\b|\brisk rating of\b"
    r"|\bwill (?:certainly |definitely |inevitably |again )?(?:occur|happen|recur|flood|be inundated|inundate)\b"
    r"|\bis (?:certain|guaranteed|bound) to\b|\bguarantee[sd]?\b|\bwithout doubt\b",
    re.I,
)
_CURRENT_STATUS = re.compile(
    r"\b(?:is|are|remains?|has been|have been|was subsequently|were subsequently)\s+"
    r"(?:now\s+|currently\s+|still\s+|since\s+)?"
    r"(?:installed|operational|in place|functioning|in service|implemented|commissioned|upgraded|strengthened)\b",
    re.I,
)
_HEDGE = re.compile(
    r"\b(?:verify|verif\w+|confirm\w*|whether|if|unknown|not documented|not known|check\w*|assess\w*|"
    r"inspect\w*|ensure|establish|determine|cannot|no evidence|not evidence|does not|do not|"
    r"recommend\w*|require\w*|not in place|was not|were not|had not|not been)\b",
    re.I,
)
_ORDER = re.compile(
    r"\b(?:it is hereby|you are hereby|hereby (?:ordered|directed)|this is an order|"
    r"by order of|is ordered to|are ordered to)\b",
    re.I,
)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(text.strip()) if s]


class _Checker:
    def __init__(self, bundle: EvidenceBundle, other_mine_names: list[str] | None):
        self.bundle = bundle
        self.report = ValidationReport(passed=True)
        self.valid_ids = set(bundle.sources)
        corpus = evidence_corpus(bundle)
        self.corpus_numbers = {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(corpus)}
        # whole numbers also count when they occur as a digit run (e.g. the 16 in "16.08.17")
        self.corpus_numbers |= {str(int(n)) for n in re.findall(r"\d+", corpus)}
        self.corpus_numbers |= set(re.findall(r"\d+", corpus))
        # counts the plan may legitimately state
        self.corpus_numbers |= {str(len(bundle.incidents)), str(len(bundle.other_incidents)),
                                str(len(bundle.findings)), str(len(bundle.incidents) + len(bundle.other_incidents)),
                                str(sum(i.persons_killed_count or 0 for i in bundle.incidents))}
        self.corpus_numbers |= {str(n) for n in range(1, 7)}  # stage numbers
        corpus_key = norm_name(corpus)
        own = norm_name(bundle.mine.mine_name)
        self.foreign = []
        for name in other_mine_names or []:
            key = norm_name(name)
            if len(key) >= 7 and " " in key and key not in corpus_key and key not in own and own not in key:
                self.foreign.append((key, name))

    def issue(self, severity: str, code: str, location: str, message: str, action: str) -> None:
        self.report.issues.append(ValidationIssue(severity=severity, code=code, location=location,
                                                  message=message, action_taken=action))

    # -- generic prose checks: returns cleaned text
    def clean_prose(self, text: str | None, location: str, *, factual: bool) -> str:
        if not text:
            return text or ""
        kept = []
        for sentence in _sentences(text):
            problem = None
            if m := _PREDICTIVE.search(sentence):
                problem = ("predictive_claim", f"Unsupported predictive wording: '{m.group(0)}'.")
            elif _ORDER.search(sentence):
                problem = ("order_language", "Wording reads as an official order.")
            elif _CURRENT_STATUS.search(sentence) and not _HEDGE.search(sentence):
                problem = ("current_status_claim",
                           "Implies a historical measure is in place today without verification.")
            else:
                key = norm_name(sentence)
                hit = next((name for k, name in self.foreign if f" {k} " in f" {key} "), None)
                if hit:
                    problem = ("other_mine_reference", f"Mentions another mine ('{hit}') not in this mine's evidence.")
                elif factual:
                    bad = [n for n in _NUMBER.findall(sentence)
                           if n.replace(",", "").rstrip(".") not in self.corpus_numbers]
                    if bad:
                        problem = ("unsupported_figure",
                                   f"Figure(s) not found in the evidence: {', '.join(sorted(set(bad)))}.")
            if problem:
                self.report.claims_removed += 1
                self.issue("error", problem[0], location, f"{problem[1]} Sentence: \"{sentence[:220]}\"",
                           "sentence removed")
            else:
                kept.append(sentence)
        return " ".join(kept)

    def clean_ids(self, ids: list[str], location: str) -> list[str]:
        good = [i for i in dict.fromkeys(ids) if i in self.valid_ids]
        for bad in [i for i in ids if i not in self.valid_ids]:
            self.issue("error", "unknown_citation", location,
                       f"Cited source '{bad}' is not in this mine's evidence bundle.", "citation removed")
        return good


def validate_plan(
    draft: PlanDraft, bundle: EvidenceBundle, other_mine_names: list[str] | None = None
) -> tuple[PlanDraft, ValidationReport]:
    """Return a repaired copy of the draft and the report of everything that was changed."""
    chk = _Checker(bundle, other_mine_names)
    report = chk.report
    has_evidence = bundle.has_water_evidence

    # ---- stages: enforce the six canonical stages in order
    by_number = {s.stage_number: s for s in draft.stages}
    if sorted(by_number) != [1, 2, 3, 4, 5, 6]:
        chk.issue("error", "stage_structure", "stages",
                  "Plan does not contain exactly the six required stages.", "none")
    stages: list[PlanStage] = []
    counter = 0
    for number, title in enumerate(STAGE_TITLES, 1):
        stage = by_number.get(number)
        if stage is None:
            continue
        if stage.title.strip() != title:
            chk.issue("warning", "stage_title", f"stage {number}",
                      f"Stage title '{stage.title}' replaced by the required title.", "title normalised")
        steps: list[PlanStep] = []
        for step in stage.steps:
            counter += 1
            loc = f"stage {number}, step {counter}"
            ids = chk.clean_ids(step.evidence_source_ids, loc)
            action = chk.clean_prose(step.action, f"{loc} (action)", factual=False)
            relevance = chk.clean_prose(step.relevance_to_this_mine, f"{loc} (relevance)", factual=False)
            evidence = chk.clean_prose(step.historical_evidence, f"{loc} (historical evidence)", factual=True) or None
            basis = step.basis
            if not has_evidence and (basis == "documented_evidence" or evidence or ids):
                chk.issue("error", "narrative_without_evidence", loc,
                          "Step claims historical support although no water-related record exists for this mine.",
                          "downgraded to general practice")
                basis, evidence, ids = "general_practice", None, []
            elif basis == "documented_evidence" and not ids:
                chk.issue("error", "uncited_evidence_step", loc,
                          "Step is marked as evidence-based but cites no valid source.",
                          "downgraded to general practice")
                basis, evidence = "general_practice", None
            elif basis == "documented_evidence" and not evidence:
                chk.issue("warning", "evidence_text_missing", loc,
                          "Evidence-based step has no surviving description of the evidence.",
                          "downgraded to general practice")
                basis, ids = "general_practice", []
            elif basis == "general_practice" and (evidence or ids):
                chk.issue("warning", "basis_mismatch", loc,
                          "Step labelled general practice carried historical evidence.",
                          "historical evidence removed")
                evidence, ids = None, []
            if len(action) < 5:
                chk.issue("error", "empty_step", loc, "Nothing supportable remained of this step.", "step removed")
                counter -= 1
                continue
            report.steps_total += 1
            report.steps_evidence_backed += basis == "documented_evidence"
            steps.append(step.model_copy(update={
                "step_number": counter, "action": action, "relevance_to_this_mine": relevance,
                "historical_evidence": evidence, "evidence_source_ids": ids, "basis": basis,
                "verify_before_acting": [chk.clean_prose(v, f"{loc} (verify)", factual=False) for v in step.verify_before_acting],
                "dependencies_and_approvals": [chk.clean_prose(v, f"{loc} (approvals)", factual=False) for v in step.dependencies_and_approvals],
            }))
        if not steps:
            chk.issue("error", "empty_stage", f"stage {number}", "Stage has no usable steps.", "none")
            continue
        stages.append(PlanStage(stage_number=number, title=title, steps=steps))

    # ---- summary and observations
    summary = chk.clean_prose(draft.historical_incident_summary, "historical summary", factual=True)
    if not has_evidence:
        summary = (
            "No water-related record for this mine was found in the processed SANKET documents. "
            "This does not establish that the mine has never experienced inundation."
        )
    observations: list[Observation] = []
    for n, obs in enumerate(draft.mine_specific_observations, 1):
        loc = f"observation {n}"
        if not has_evidence:
            chk.issue("error", "narrative_without_evidence", loc,
                      "Mine-specific observation offered without any water-related record.", "observation removed")
            continue
        ids = chk.clean_ids(obs.evidence_source_ids, loc)
        text = chk.clean_prose(obs.text, loc, factual=obs.kind == "documented_fact")
        kind = obs.kind
        if kind == "documented_fact" and not ids:
            chk.issue("warning", "uncited_fact", loc, "Observation labelled as documented fact has no valid citation.",
                      "relabelled as inference")
            kind = "reasonable_inference"
        if text:
            observations.append(Observation(text=text, kind=kind, evidence_source_ids=ids))

    def clean_list(items: list[str], name: str) -> list[str]:
        out = [chk.clean_prose(x, name, factual=False) for x in items]
        return [x for x in dict.fromkeys(out) if x]

    cleaned = PlanDraft.model_construct(
        historical_incident_summary=summary,
        mine_specific_observations=observations,
        stages=stages,
        assumptions_and_uncertainties=clean_list(draft.assumptions_and_uncertainties, "assumptions"),
        missing_information=clean_list(draft.missing_information, "missing information"),
        required_human_verification_and_approvals=clean_list(
            draft.required_human_verification_and_approvals, "verification and approvals"),
    )
    report.passed = len(stages) == 6 and not any(i.action_taken == "none" for i in report.issues)
    return cleaned, report
