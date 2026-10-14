"""Turn raw extracted records into the normalised mine / incident / finding / source schema.

Design rules, all aimed at never manufacturing facts:

* Mine identity is decided by structured fields only. A mine described in the yearly
  fatal-accident summaries is identified by (category, name, state, district). A mine that
  appears only in the historic tables is identified by (category, name) because those
  tables print nothing else - this is recorded as ``identity_basis="name_only"``.
* Two records become one incident only when they are in the same category, carry the same
  date and have matching mine names (a table row and its court-of-inquiry row, or a table
  row and its narrative summary).
* Spelling variants are merged only through the curated alias file. Automatically detected
  look-alikes are reported for review, not merged.
* Original text is always preserved next to any normalised value.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from rapidfuzz import fuzz

from .pdf_extractor import ExtractionResult
from .schemas import Incident, Mine, PreventiveFinding, SourceReference

log = logging.getLogger(__name__)

CATEGORY_PREFIX = {"coal": "coal", "metal_non_coal": "mnc"}


@dataclass
class NormalizedDataset:
    mines: list[Mine] = field(default_factory=list)
    incidents: list[Incident] = field(default_factory=list)
    findings: list[PreventiveFinding] = field(default_factory=list)
    sources: list[SourceReference] = field(default_factory=list)
    merges: list[dict] = field(default_factory=list)
    duplicate_candidates: list[dict] = field(default_factory=list)
    aliases_applied: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- text helpers
def norm_name(text: str | None) -> str:
    """Case/punctuation-insensitive key. Deliberately conservative: 'OC-II' != 'OC-III'."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", (text or "").upper())).strip()


def slugify(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def make_mine_id(category: str, name_key: str, state_key: str | None, district_key: str | None) -> str:
    digest = hashlib.sha1(
        "|".join([category, name_key, state_key or "", district_key or ""]).encode("utf-8")
    ).hexdigest()[:8]
    return f"{CATEGORY_PREFIX[category]}-{slugify(name_key)[:40]}-{digest}"


def load_aliases(path: Path | None) -> dict[str, dict[str, str]]:
    """Curated spelling-variant map: {category: {VARIANT KEY: CANONICAL KEY}}."""
    if path is None or not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        cat: {norm_name(k): norm_name(v) for k, v in mapping.items()}
        for cat, mapping in data.items()
        if isinstance(mapping, dict)
    }


# --------------------------------------------------------------------------- dates
class CenturyTracker:
    """Resolve two-digit years in a table that is printed in chronological order."""

    def __init__(self, start_century: int = 1900):
        self.century = start_century
        self.last_yy: int | None = None

    def resolve(self, yy: int) -> int:
        if self.last_yy is not None and yy < self.last_yy - 50:
            self.century += 100
            self.last_yy = yy
        elif self.last_yy is None or yy > self.last_yy:
            self.last_yy = yy
        return self.century + yy


def parse_source_date(
    text: str, *, two_digit_base: int | None = None, tracker: CenturyTracker | None = None
) -> tuple[str | None, int | None, str, tuple[int, int] | None]:
    """Return (iso_date, year, quality, (day, month)) for a date printed as dd<sep>mm<sep>yy[yy].

    ``iso_date`` is None whenever the printed date cannot be read unambiguously.
    """
    m = re.fullmatch(r"\s*(\d{1,2})([-/.])(\d{1,2})[-/.](\d{2,4})\s*", text or "")
    if not m:
        return None, None, "unparsed", None
    day, sep, month, year_text = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
    quality = "exact"
    if len(year_text) == 4:
        year = int(year_text)
    elif len(year_text) == 2 and two_digit_base is not None:
        year = two_digit_base + int(year_text)
    elif len(year_text) == 2 and tracker is not None:
        year = tracker.resolve(int(year_text))
        quality = "century_inferred"
        # Hyphenated two-digit dates in the non-coal table look like spreadsheet-reformatted
        # cells; day and month cannot be told apart when both are <= 12.
        if sep == "-" and day <= 12 and month <= 12 and day != month:
            return None, year, "day_month_ambiguous", (day, month)
    else:
        return None, None, "unparsed", None
    try:
        return date(year, month, day).isoformat(), year, quality, (day, month)
    except ValueError:
        return None, year, "unparsed", None


# --------------------------------------------------------------------------- classification
_IRRUPTION = re.compile(r"irruption|inundation", re.I)
_DROWNING = re.compile(r"drown", re.I)
_WATER_NARRATIVE = re.compile(
    r"\b(drown\w*|inrush\w*|inundat\w*|irruption|water[- ]?logged|flood\w*|submerg\w*|"
    r"water body|water current|accumulated (?:surface |rain )?water|sump)\b",
    re.I,
)


def classify_water(cause_code: str | None, cause_text: str | None, narrative: str | None) -> tuple[str, str | None]:
    """Classify how a record relates to water and state the documented basis."""
    cause_text = cause_text or ""
    if cause_code == "0991" or _IRRUPTION.search(cause_text):
        return "irruption", f"Source classifies the cause as '{cause_text.strip()}'" + (
            f" (DGMS code {cause_code})" if cause_code else ""
        )
    if cause_code == "0993" or _DROWNING.search(cause_text):
        return "drowning", f"Source classifies the cause as '{cause_text.strip()}'" + (
            f" (DGMS code {cause_code})" if cause_code else ""
        )
    if narrative:
        terms = sorted({m.group(0).lower() for m in _WATER_NARRATIVE.finditer(narrative)})
        if terms:
            return "water_keyword", (
                "Keyword match in the DGMS narrative (" + ", ".join(terms) + "); the source "
                f"classifies the cause as '{cause_text.strip() or 'not stated'}'"
            )
    return "none", None


_MEASUREMENT = re.compile(
    r"(?:about |approximately |nearly |around )?\d+(?:\.\d+)?\s?"
    r"(?:cubic met\w+|cu\.?\s?m\b|m3\b|m\b|metres?\b|meters?\b|km\b|cm\b|mm\b|feet\b|ft\b|"
    r"litres?\b|kilolitres?\b|KL\b|hours?\b|hrs?\b|minutes?\b)"
    r"(?:\s(?:thick|high|deep|long|wide|height|depth|in height|in depth))?",
    re.I,
)
_REGULATION = re.compile(
    r"(?:Regulations?|Reg\.|Rules?|Sections?)\s*\d+[A-Z]?(?:\s?\([0-9A-Za-z]+\))*"
    r"|D\.?\s?G\.?(?:M\.?S\.?)?'?s?\.?\s*(?:\(?Tech(?:nical|\.)?\)?\s*)?Circular[^;,\n]{0,40}"
    r"|(?:Coal Mines|Metalliferous Mines|Oil Mines) Regulations?,?\s*\d{4}"
    r"|Mines Act,?\s*\d{4}|Mines Rules,?\s*\d{4}|\b(?:CMR|MMR|OMR),?\s*\d{4}",
    re.I,
)


def measurements_in(text: str | None) -> list[str]:
    seen: list[str] = []
    for m in _MEASUREMENT.finditer(text or ""):
        value = " ".join(m.group(0).split())
        if value not in seen:
            seen.append(value)
    return seen


def regulations_in(text: str | None) -> list[str]:
    seen: list[str] = []
    for m in _REGULATION.finditer(text or ""):
        value = " ".join(m.group(0).split()).rstrip(" .")
        if value not in seen:
            seen.append(value)
    return seen


# --------------------------------------------------------------------------- findings
_AVERTED = re.compile(
    r"[,;.\s]*(?:then\s+)?(?:this|the)\s+accident\s+(?:could|would|might|may)\s+have\s+been\s+"
    r"(?:averted|avoided|prevented)\W*$",
    re.I,
)
_HAD_PREFIX = re.compile(r"^\s*had\b\s*,?\s*", re.I)
_ENUM = re.compile(r"^\(?(?:[ivx]{1,5}|[IVX]{1,5}|\d{1,2})[\).]\s+|^\(?[a-h]\)\s+")

# Ordered keyword rules. Purely a display aid: a category is null when nothing matches.
_CATEGORY_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("Water barrier / dam / stopping integrity",
     re.compile(r"\bdam\b|retaining wall|stopping|barrier|parting|embankment|\bbund\b|sludge", re.I)),
    ("Withdrawal of persons / danger level",
     re.compile(r"withdraw|danger level|water[- ]?logged|persons? (?:been )?not (?:been )?(?:sent|allowed|deployed)", re.I)),
    ("Pumping / dewatering arrangement", re.compile(r"pump|dewater|sump|pontoon", re.I)),
    ("Fencing / guarding of water bodies", re.compile(r"fenc|guard|barricad|parapet|berm", re.I)),
    ("Life-saving equipment", re.compile(r"life[- ]?(?:line|jacket|buoy)|safety belt", re.I)),
    ("Risk assessment / SOP / code of practice",
     re.compile(r"risk assessment|\bSOP\b|standard operating|code of (?:safe )?practice|safety management plan", re.I)),
    ("Survey / plans / old workings", re.compile(r"survey|\bplans?\b|old workings?|bore ?holes?", re.I)),
    ("Supervision / competent person",
     re.compile(r"supervis|competent person|manager|overman|sirdar|official", re.I)),
    ("Training / awareness", re.compile(r"train|awareness", re.I)),
)


def categorize_finding(text: str) -> str | None:
    for label, pattern in _CATEGORY_RULES:
        if pattern.search(text):
            return label
    return None


def split_findings(block: str | None) -> tuple[list[str], str]:
    """Split a 'Had, ... this accident could have been averted' block into its items.

    Returns (items, quality). Items keep the source wording; only line wrapping is undone.
    """
    if not block or not block.strip():
        return [], "itemised"
    lines = block.splitlines()
    lines[0] = _HAD_PREFIX.sub("", lines[0])
    items: list[list[str]] = []
    enumerated = any(_ENUM.match(line.strip()) for line in lines)
    blank_before = True
    for raw in lines:
        line = raw.strip()
        if not line:
            blank_before = True
            continue
        new_item = bool(_ENUM.match(line)) if enumerated else blank_before
        if new_item or not items:
            items.append([line])
        else:
            items[-1].append(line)
        blank_before = False
    texts = []
    for parts in items:
        text = _AVERTED.sub("", re.sub(r"\s+", " ", " ".join(parts)).strip()).strip(" ,;")
        text = _ENUM.sub("", text).strip()
        if len(text) >= 12:
            texts.append(text)
    quality = "itemised" if (enumerated or len(texts) > 1) else "unsplit_block"
    return texts, quality


# --------------------------------------------------------------------------- normalisation
def _names_match(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a == b or fuzz.token_set_ratio(a, b) >= 82)


def _verification(warnings: list[str]) -> str:
    return "needs_review" if warnings else "parsed_clean"


def normalize(results: list[ExtractionResult], aliases: dict[str, dict[str, str]] | None = None) -> NormalizedDataset:
    aliases = aliases or {}
    ds = NormalizedDataset()
    protos: list[dict] = []  # incident prototypes before mine assignment

    def add_source(source_id, filename, pages, section, text, method, warnings) -> None:
        ds.sources.append(
            SourceReference(
                source_id=source_id, source_filename=filename, page_start=min(pages),
                page_end=max(pages), pages=sorted(set(pages)), section=section,
                record_type="incident", source_text=text, extraction_method=method,
                verification_status=_verification(warnings), warnings=list(warnings),
            )
        )

    for res in results:
        cat = res.document.category
        if cat not in CATEGORY_PREFIX:
            continue
        prefix, filename = CATEGORY_PREFIX[cat], res.document.filename

        # ---- yearly fatal-accident summaries (2016-2022 in the supplied volumes)
        for rec in res.fatal_records:
            warnings = list(rec.warnings)
            iso, year, quality, dm = parse_source_date(rec.date_text, two_digit_base=2000)
            if year is not None and year != rec.year:
                warnings.append(
                    f"Printed date {rec.date_text!r} is not in the chapter year {rec.year}; "
                    "the chapter year is used and the date is left unparsed."
                )
                iso, year, quality = None, rec.year, "unparsed"
            if year is None:
                warnings.append(f"Date {rec.date_text!r} could not be parsed.")
                year = rec.year
            incident_id = f"{prefix}-{rec.year}-F{rec.serial:03d}"
            source_id = f"S-{incident_id}"
            add_source(source_id, filename, rec.pages, "fatal_accident_summary", rec.source_text,
                       "pymupdf_text_record_parser", warnings)
            relevance, basis = classify_water(rec.cause_code, rec.cause_label, rec.narrative)
            protos.append({
                "incident_id": incident_id, "category": cat, "kind": "fatal",
                "mine_name": rec.mine_name, "state": rec.state, "district": rec.district,
                "owner": rec.owner, "iso": iso, "year": year, "dm": dm, "date_text": rec.date_text,
                "quality": quality, "time_text": rec.time_text,
                "sections": ["fatal_accident_summary"], "cause_code": rec.cause_code,
                "cause_label": rec.cause_label, "cause_group": rec.parent_label,
                "relevance": relevance, "basis": basis, "description": rec.narrative or None,
                "findings_text": rec.findings_text or None,
                "killed": len(rec.persons_killed) or None, "injured": None,
                "persons": list(rec.persons_killed), "court": None,
                "source_ids": [source_id], "warnings": warnings,
            })

        # ---- major accidents since 1901
        tracker = CenturyTracker()
        table_protos: list[dict] = []
        for row in res.major_rows:
            warnings = list(row.warnings)
            iso, year, quality, dm = parse_source_date(row.date_text, tracker=tracker)
            if quality == "day_month_ambiguous":
                warnings.append(
                    f"Date {row.date_text!r}: day and month cannot be told apart and the century "
                    "is inferred from the table order; only the year is used."
                )
            elif quality == "century_inferred":
                warnings.append(f"Date {row.date_text!r}: century inferred from the table order.")
            elif quality == "unparsed":
                warnings.append(f"Date {row.date_text!r} could not be parsed.")
            incident_id = f"{prefix}-MAJ-{row.serial:03d}"
            source_id = f"S-{incident_id}"
            add_source(source_id, filename, [row.page], "major_accident_table", row.source_text,
                       "pymupdf_cell_per_line_table_parser", warnings)
            relevance, basis = classify_water(None, row.cause, None)
            table_protos.append({
                "incident_id": incident_id, "category": cat, "kind": "major",
                "mine_name": row.mine_name, "state": None, "district": None, "owner": None,
                "iso": iso, "year": year, "dm": dm, "date_text": row.date_text, "quality": quality,
                "time_text": None, "sections": ["major_accident_table"], "cause_code": None,
                "cause_label": row.cause or None, "cause_group": None, "relevance": relevance,
                "basis": basis, "description": None, "findings_text": None, "killed": row.killed,
                "injured": row.seriously_injured, "persons": [], "court": None,
                "source_ids": [source_id], "warnings": warnings,
            })

        # ---- courts of inquiry: attach to the matching major-accident row when there is one
        tracker = CenturyTracker()
        for row in res.court_rows:
            warnings = list(row.warnings)
            iso, year, quality, dm = parse_source_date(row.date_text, tracker=tracker)
            if quality == "century_inferred":
                warnings.append(f"Date {row.date_text!r}: century inferred from the table order.")
            elif quality != "exact":
                warnings.append(f"Date {row.date_text!r} could not be parsed unambiguously.")
            suffix = f"{row.serial or 0:02d}{row.sub_item or ''}"
            source_id = f"S-{prefix}-COI-{suffix}"
            add_source(source_id, filename, [row.page], "court_of_inquiry_table", row.source_text,
                       "pymupdf_table_parser", warnings)
            key = norm_name(row.mine_name)
            target = next(
                (p for p in table_protos
                 if p["year"] == year and p["dm"] == dm and dm is not None
                 and _names_match(norm_name(p["mine_name"]), key)),
                None,
            )
            court_text = f"Court of inquiry instituted ({row.cause or 'cause not stated'}). " + (
                row.court_text or ""
            )
            if target is not None:
                target["sections"].append("court_of_inquiry_table")
                target["source_ids"].append(source_id)
                target["court"] = court_text.strip()
                if row.killed is not None and target["killed"] is not None and row.killed != target["killed"]:
                    target["warnings"].append(
                        f"Persons killed differ between tables: {target['killed']} (major accidents) "
                        f"vs {row.killed} (court of inquiry)."
                    )
                ds.merges.append({"kept": target["incident_id"], "merged_source": source_id,
                                  "basis": "same category, same date, matching mine name"})
                continue
            relevance, basis = classify_water(None, row.cause, None)
            table_protos.append({
                "incident_id": f"{prefix}-COI-{suffix}", "category": cat, "kind": "court",
                "mine_name": row.mine_name, "state": None, "district": None, "owner": None,
                "iso": iso, "year": year, "dm": dm, "date_text": row.date_text, "quality": quality,
                "time_text": None, "sections": ["court_of_inquiry_table"], "cause_code": None,
                "cause_label": row.cause, "cause_group": None, "relevance": relevance,
                "basis": basis, "description": None, "findings_text": None, "killed": row.killed,
                "injured": None, "persons": [], "court": court_text.strip(),
                "source_ids": [source_id], "warnings": warnings,
            })

        # ---- a table row and a narrative summary describing the same accident
        fatal_here = [p for p in protos if p["category"] == cat and p["kind"] == "fatal"]
        for tp in table_protos:
            target = next(
                (p for p in fatal_here
                 if p["year"] == tp["year"] and p["dm"] == tp["dm"] and tp["dm"] is not None
                 and _names_match(norm_name(p["mine_name"]), norm_name(tp["mine_name"]))),
                None,
            )
            if target is None:
                protos.append(tp)
                continue
            target["sections"] += tp["sections"]
            target["source_ids"] += tp["source_ids"]
            target["court"] = target["court"] or tp["court"]
            target["injured"] = tp["injured"] if target["injured"] is None else target["injured"]
            if tp["killed"] is not None and target["killed"] not in (None, tp["killed"]):
                target["warnings"].append(
                    f"Persons killed differ between sections: {target['killed']} named in the "
                    f"summary vs {tp['killed']} in the table."
                )
            if target["relevance"] == "none" and tp["relevance"] != "none":
                target["relevance"], target["basis"] = tp["relevance"], tp["basis"]
            ds.merges.append({"kept": target["incident_id"], "merged_source": tp["source_ids"][0],
                              "basis": "same category, same date, matching mine name"})

    # ---- mine assignment (structured keys only)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in protos:
        key = norm_name(p["mine_name"])
        canonical = aliases.get(p["category"], {}).get(key, key)
        if canonical != key:
            ds.aliases_applied.append({"category": p["category"], "variant": p["mine_name"],
                                       "canonical_key": canonical, "incident_id": p["incident_id"]})
        located = p["kind"] == "fatal" and (p["state"] or p["district"])
        gkey = (p["category"], canonical,
                norm_name(p["state"]) if located else None,
                norm_name(p["district"]) if located else None)
        groups[gkey].append(p)

    by_name: dict[tuple, list[str]] = defaultdict(list)
    for (cat, name_key, state_key, district_key), items in groups.items():
        mine_id = make_mine_id(cat, name_key, state_key, district_key)
        by_name[(cat, name_key)].append(mine_id)
        names = Counter(p["mine_name"] for p in items)
        owners = Counter(p["owner"] for p in items if p["owner"])
        years = [p["year"] for p in items if p["year"]]
        rel = Counter(p["relevance"] for p in items)
        located = state_key is not None or district_key is not None
        first = items[0]
        ds.mines.append(Mine(
            mine_id=mine_id, mine_name=names.most_common(1)[0][0], mine_category=cat,
            name_variants=sorted(n for n in names if n != names.most_common(1)[0][0]),
            state=first["state"] if located else None,
            district=first["district"] if located else None,
            operator=owners.most_common(1)[0][0] if owners else None,
            operator_variants=sorted(o for o in owners if o != owners.most_common(1)[0][0]),
            identity_basis="name_state_district" if located else "name_only",
            incident_count=len(items),
            water_incident_count=len(items) - rel["none"],
            irruption_count=rel["irruption"], drowning_count=rel["drowning"],
            water_keyword_count=rel["water_keyword"],
            earliest_year=min(years) if years else None, latest_year=max(years) if years else None,
        ))
        for p in items:
            texts, quality = split_findings(p["findings_text"])
            ds.incidents.append(Incident(
                incident_id=p["incident_id"], mine_id=mine_id, mine_name_in_source=p["mine_name"],
                incident_date=p["iso"], incident_year=p["year"], date_text=p["date_text"],
                date_quality=p["quality"], time_text=p["time_text"], record_sections=p["sections"],
                cause_code=p["cause_code"], cause_label=p["cause_label"],
                cause_group=p["cause_group"], water_relevance=p["relevance"],
                water_basis=p["basis"], description=p["description"],
                findings_text=p["findings_text"], persons_killed_count=p["killed"],
                seriously_injured_count=p["injured"], persons_killed=p["persons"],
                measurements_mentioned=measurements_in(p["description"]),
                regulations_cited=regulations_in(p["findings_text"]),
                court_of_inquiry=p["court"], source_ids=p["source_ids"], warnings=p["warnings"],
            ))
            for n, text in enumerate(texts, 1):
                category = categorize_finding(text)
                ds.findings.append(PreventiveFinding(
                    measure_id=f"{p['incident_id']}-M{n:02d}", mine_id=mine_id,
                    incident_id=p["incident_id"], measure_text=text, measure_category=category,
                    category_method="keyword_rule" if category else None,
                    regulations_cited=regulations_in(text), extraction_quality=quality,
                    source_ids=[p["source_ids"][0]],
                ))

    # same printed name, different identity -> surface as "possibly related", never merged
    related = {mid: [o for o in ids if o != mid] for ids in by_name.values() if len(ids) > 1 for mid in ids}
    for mine in ds.mines:
        mine.possibly_related_mine_ids = related.get(mine.mine_id, [])

    ds.duplicate_candidates = _duplicate_candidates(ds.mines)
    ds.mines.sort(key=lambda m: (m.mine_category, m.mine_name.upper(), m.state or "", m.district or ""))
    ds.incidents.sort(key=lambda i: (i.mine_id, i.incident_year or 0, i.incident_date or "", i.incident_id))
    _validate(ds)
    log.info("Normalised %d mines, %d incidents, %d findings, %d source references",
             len(ds.mines), len(ds.incidents), len(ds.findings), len(ds.sources))
    return ds


def _duplicate_candidates(mines: list[Mine]) -> list[dict]:
    """Look-alike mine names that were NOT merged. Listed for human review."""
    out: list[dict] = []
    by_cat: dict[str, list[Mine]] = defaultdict(list)
    for m in mines:
        by_cat[m.mine_category].append(m)
    for items in by_cat.values():
        keyed = [(norm_name(m.mine_name), m) for m in items]
        for i, (ka, a) in enumerate(keyed):
            for kb, b in keyed[i + 1 :]:
                same_place = (a.state is None or b.state is None
                              or norm_name(a.state) == norm_name(b.state))
                if not same_place:
                    continue
                if ka == kb:
                    reason = "identical name, different or undocumented location"
                elif abs(len(ka) - len(kb)) <= 3 and fuzz.ratio(ka, kb) >= 90:
                    reason = f"similar spelling (similarity {fuzz.ratio(ka, kb):.0f})"
                else:
                    continue
                out.append({
                    "category": a.mine_category, "reason": reason,
                    "mine_a": {"mine_id": a.mine_id, "name": a.mine_name, "state": a.state, "district": a.district},
                    "mine_b": {"mine_id": b.mine_id, "name": b.mine_name, "state": b.state, "district": b.district},
                })
    return out


def _validate(ds: NormalizedDataset) -> None:
    """Schema-level integrity checks; problems are reported, never silently repaired."""
    mine_ids = {m.mine_id for m in ds.mines}
    source_ids = {s.source_id for s in ds.sources}
    incident_ids = [i.incident_id for i in ds.incidents]
    if len(mine_ids) != len(ds.mines):
        ds.warnings.append("Duplicate mine_id values were generated.")
    if len(set(incident_ids)) != len(incident_ids):
        ds.warnings.append("Duplicate incident_id values were generated.")
    if len(source_ids) != len(ds.sources):
        ds.warnings.append("Duplicate source_id values were generated.")
    for inc in ds.incidents:
        if inc.mine_id not in mine_ids:
            ds.warnings.append(f"{inc.incident_id}: unknown mine_id {inc.mine_id}.")
        if not inc.source_ids or any(s not in source_ids for s in inc.source_ids):
            ds.warnings.append(f"{inc.incident_id}: missing source reference.")
    known = set(incident_ids)
    for f in ds.findings:
        if f.incident_id not in known:
            ds.warnings.append(f"{f.measure_id}: unknown incident_id {f.incident_id}.")
    used = {s for i in ds.incidents for s in i.source_ids}
    orphans = source_ids - used
    if orphans:
        ds.warnings.append(f"{len(orphans)} source reference(s) are not linked to any incident.")
