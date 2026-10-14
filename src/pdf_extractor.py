"""Structure-aware extraction of the DGMS SANKET volumes.

The two source PDFs are born-digital (Microsoft Word) statistical volumes published by the
Directorate General of Mines Safety. Inspection showed three record-bearing structures that
carry mine-level information, and this module has one parser for each:

1. "Summary of Findings of Fatal Accidents during the year YYYY" - numbered free-text
   records with a fixed header (Date / Mine / Time / Owner / Dist. / State / Person(s)
   Killed), a narrative, and a "Had ... this accident could have been averted" findings
   block, grouped under DGMS cause-code headings.
2. "Major accident since 1901" - a table that PyMuPDF yields one cell per line.
3. "Court of Inquiry" - a table (fixed-width text in the coal volume, one cell per line in
   the non-coal volume).

Nothing is inferred here beyond what the text states: every record keeps its verbatim
source text and the 1-based PDF page numbers it was read from, and anything the parser
could not place is reported as a warning instead of being guessed.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

log = logging.getLogger(__name__)

CATEGORY_COAL = "coal"
CATEGORY_NON_COAL = "metal_non_coal"


class PdfExtractionError(RuntimeError):
    """Raised when a PDF cannot be opened or contains no usable text."""


# --------------------------------------------------------------------------- data classes
@dataclass
class PageInfo:
    page_number: int  # 1-based, as shown by a PDF viewer
    char_count: int
    image_count: int
    kind: str  # text | image_only | blank
    lines: list[str] = field(repr=False, default_factory=list)


@dataclass
class PdfDocument:
    path: Path
    filename: str
    sha256: str
    page_count: int
    category: str | None
    title: str | None
    pages: list[PageInfo] = field(repr=False, default_factory=list)

    @property
    def image_only_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.kind == "image_only"]

    @property
    def blank_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.kind == "blank"]


@dataclass
class RawFatalRecord:
    filename: str
    category: str
    year: int
    serial: int
    date_text: str
    mine_name: str
    time_text: str | None = None
    owner: str | None = None
    district: str | None = None
    state: str | None = None
    persons_killed: list[str] = field(default_factory=list)
    narrative: str = ""
    findings_text: str = ""
    cause_code: str | None = None
    cause_label: str | None = None
    parent_code: str | None = None
    parent_label: str | None = None
    pages: list[int] = field(default_factory=list)
    source_text: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class RawMajorRow:
    filename: str
    category: str
    serial: int
    date_text: str
    mine_name: str
    killed: int | None
    seriously_injured: int | None
    cause: str
    page: int
    source_text: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class RawCourtRow:
    filename: str
    category: str
    serial: int | None
    sub_item: str | None
    date_text: str
    mine_name: str
    killed: int | None
    cause: str | None
    court_text: str
    page: int
    source_text: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    document: PdfDocument
    fatal_records: list[RawFatalRecord] = field(default_factory=list)
    major_rows: list[RawMajorRow] = field(default_factory=list)
    court_rows: list[RawCourtRow] = field(default_factory=list)
    stated_fatal_totals: dict[int, int] = field(default_factory=dict)
    section_pages: dict[str, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- page loading
_RUNNING_HEADER = re.compile(
    r"^Statistical Analysis & Numbers: Key Evaluation of Trends in (Non-)?Coal Mine Accidents$"
)
_NON_COAL_TITLE = re.compile(r"Trends\s+in\s+Non-?\s?Coal\s+Mine\s+Accidents", re.I)
_COAL_TITLE = re.compile(r"Trends\s+in\s+Coal\s+Mine\s+Accidents", re.I)


def _strip_page_furniture(lines: list[str]) -> list[str]:
    """Remove the page number and running header from the top of a page."""
    out, seen = [], 0
    for line in lines:
        s = line.strip()
        if seen < 5 and s:
            seen += 1
            if s.isdigit() or s == "SANKET" or _RUNNING_HEADER.match(s):
                continue
        out.append(line.rstrip())
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_pdf(path: Path) -> PdfDocument:
    """Open a PDF read-only and classify every page. The file is never modified."""
    path = Path(path)
    if not path.is_file():
        raise PdfExtractionError(f"PDF not found: {path}")
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # pymupdf raises several unrelated exception types
        raise PdfExtractionError(f"Cannot open {path.name}: {exc}") from exc
    try:
        if doc.needs_pass:
            raise PdfExtractionError(f"{path.name} is password protected.")
        pages: list[PageInfo] = []
        head_text = ""
        for index, page in enumerate(doc):
            text = page.get_text("text")
            images = len(page.get_images(full=True))
            body = text.strip()
            if index < 40:
                head_text += " " + " ".join(body.split())
            if len(body) < 40 and "intentionally left blank" not in body.lower():
                kind = "image_only" if images else "blank"
            elif "intentionally left blank" in body.lower():
                kind = "blank"
            else:
                kind = "text"
            pages.append(
                PageInfo(index + 1, len(body), images, kind, _strip_page_furniture(text.splitlines()))
            )
        page_count = doc.page_count
    finally:
        doc.close()

    if not any(p.kind == "text" for p in pages):
        raise PdfExtractionError(
            f"{path.name} has no selectable text on any page (scanned document). "
            "OCR is required before it can be ingested."
        )
    if _NON_COAL_TITLE.search(head_text):
        category, title = CATEGORY_NON_COAL, "SANKET - Trends in Non-Coal Mine Accidents"
    elif _COAL_TITLE.search(head_text):
        category, title = CATEGORY_COAL, "SANKET - Trends in Coal Mine Accidents"
    else:
        category, title = None, None
    return PdfDocument(path, path.name, file_sha256(path), page_count, category, title, pages)


# --------------------------------------------------------------------------- fatal summaries
_SUMMARY_HEAD = re.compile(r"Summary of Findings of Fatal Accidents", re.I)
_SECTION_END = re.compile(r"ACCIDENT ANALYSIS IN|MAJOR ACCIDENT SINCE 1901")
_YEAR = re.compile(r"\b(19\d\d|20\d\d)\b")
_REC_START = re.compile(r"^\s*(\d{1,3})\.\s+Date\s*-\s*(\S+)\s+Mine\s*-\s*(.*?)\s*$")
_TIME = re.compile(r"Time\s*-\s*(\S+)")
_OWNER = re.compile(r"Owner\s*-\s*(.*?)\s*$")
_DIST = re.compile(r"Dist\.\s*-\s*(.*?)\s*,\s*State\s*-\s*(.*?)\s*$")
_PERSONS = re.compile(r"Person\(s\)\s+Killed\s*:\s*(.*)$")
_PERSON_LINE = re.compile(r"^\s+\d{1,2}\.(\s|$)")
_CODE = re.compile(r"^\s*Code\s*:\s*(\d{4})\s+(.*?)\s*$")
_RULE = re.compile(r"^\s*-{8,}\s*$")
_DEATHS = re.compile(r"^\s*\(\s*\d+\s+Deaths?\s*\)\s*$", re.I)
_FINDINGS_START = re.compile(r"^\s*(Had\b|had\s*,?\s*$)")
_CHAPTER_NUMBER = re.compile(r"^\s*\d{1,2}\.\s*$")
_STATED_TOTAL = re.compile(
    r"number of fatal accidents in (?:non[- ]?coal|coal) mines in the year (\d{4}) stands at\s*(\d+)",
    re.I,
)


class _FatalBuilder:
    def __init__(self, doc: PdfDocument, year: int, serial: int, date_text: str, mine: str):
        self.rec = RawFatalRecord(doc.filename, doc.category or "", year, serial, date_text, mine)
        self.state = "header"
        self.awaiting_state = False
        self.raw: list[str] = []
        self.raw_pages: list[int] = []
        self.narrative: list[str] = []
        self.findings: list[str] = []

    def feed(self, page: int, line: str) -> None:
        rec = self.rec
        self.raw.append(line)
        self.raw_pages.append(page)
        stripped = line.strip()

        if self.state == "header":
            matched = False
            if m := _PERSONS.search(line):
                self.state = "persons"
                if m.group(1).strip():
                    rec.persons_killed.append(m.group(1).strip())
                return
            if self.awaiting_state and stripped:
                # "Dist. - Kalaburagi(...),  State - " wrapped: the state is on its own line.
                rec.state, self.awaiting_state = stripped, False
                return
            if m := _DIST.search(line):
                rec.district, rec.state = m.group(1).strip() or None, m.group(2).strip() or None
                self.awaiting_state = rec.state is None
                matched = True
            else:
                # "Time - 7.15   Owner - Central Coalfields Ltd." share one line.
                if m := _OWNER.search(line):
                    rec.owner = m.group(1).strip() or None
                    matched = True
                if m := _TIME.search(line):
                    rec.time_text = m.group(1).strip()
                    matched = True
            if matched or not stripped:
                return
            if _FINDINGS_START.match(line):
                self.state = "findings"
                self.findings.append(stripped)
            elif line[:1].strip():  # text starting at the left margin = narrative
                self.state = "narrative"
                self.narrative.append(stripped)
                rec.warnings.append("No 'Person(s) Killed' line found before the narrative.")
            else:
                rec.warnings.append(f"Unrecognised header line: {stripped!r}")
            return

        if self.state == "persons":
            if not stripped:
                return
            if _FINDINGS_START.match(line):
                self.state = "findings"
                self.findings.append(stripped)
            elif _PERSON_LINE.match(line):
                rec.persons_killed.append(stripped)
            elif (
                rec.persons_killed
                and len(stripped) <= 45
                and not rec.persons_killed[-1].rstrip(". ").endswith("Years")
            ):
                # entries wrap: "1." / "S.Ahamed" / "Nijauddin,Explosive" / ... / "Years"
                rec.persons_killed[-1] += " " + stripped
            else:
                self.state = "narrative"
                self.narrative.append(stripped)
            return

        if self.state == "narrative":
            if _FINDINGS_START.match(line):
                self.state = "findings"
                self.findings.append(stripped)
            else:
                self.narrative.append(stripped)
            return

        self.findings.append(stripped)

    def finish(self) -> RawFatalRecord:
        rec = self.rec
        # The last record of a year is followed by blank divider pages and the next chapter's
        # number ("4."); neither belongs to the record or to its page provenance.
        while self.raw and (not self.raw[-1].strip() or _CHAPTER_NUMBER.match(self.raw[-1])):
            self.raw.pop()
            self.raw_pages.pop()
        for block in (self.findings, self.narrative):
            while block and (not block[-1] or _CHAPTER_NUMBER.match(block[-1])):
                block.pop()
        rec.pages = sorted(set(self.raw_pages))
        rec.narrative = _join_paragraphs(self.narrative)
        rec.findings_text = _join_block(self.findings)
        rec.source_text = "\n".join(self.raw).strip("\n")
        if not rec.mine_name:
            rec.warnings.append("Mine name is empty in the source record.")
        if not rec.narrative:
            rec.warnings.append("No narrative text was found for this record.")
        if rec.district is None or rec.state is None:
            rec.warnings.append("District/State line not found.")
        if rec.owner is None:
            rec.warnings.append("Owner line not found.")
        if not rec.persons_killed:
            rec.warnings.append("No persons-killed entries were parsed.")
        return rec


def _join_paragraphs(lines: list[str]) -> str:
    """Re-flow hard-wrapped lines into paragraphs separated by blank lines."""
    paragraphs, current = [], []
    for line in lines:
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(re.sub(r"\s+", " ", p).strip() for p in paragraphs)


def _join_block(lines: list[str]) -> str:
    """Keep line structure (enumerated findings) but trim surrounding blank lines."""
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_fatal_summaries(doc: PdfDocument, result: ExtractionResult) -> None:
    year: int | None = None
    awaiting_year = 0
    builder: _FatalBuilder | None = None
    code = label = parent_code = parent_label = None
    section_pages: list[int] = []

    def close() -> None:
        nonlocal builder
        if builder is not None:
            result.fatal_records.append(builder.finish())
            builder = None

    for page in doc.pages:
        for line in page.lines:
            if m := _STATED_TOTAL.search(line):
                result.stated_fatal_totals[int(m.group(1))] = int(m.group(2))
            if _SUMMARY_HEAD.search(line):
                close()
                awaiting_year, year = 4, None
                code = label = parent_code = parent_label = None
            if awaiting_year:
                awaiting_year -= 1
                if m := _YEAR.search(line):
                    year, awaiting_year = int(m.group(1)), 0
                continue
            if year is None:
                continue
            if _SECTION_END.search(line):
                close()
                year = None
                continue
            if page.page_number not in section_pages:
                section_pages.append(page.page_number)

            if m := _CODE.match(line):
                close()
                c, text = m.group(1), m.group(2).strip()
                if c.endswith("00"):
                    parent_code, parent_label, code, label = c, text, c, text
                else:
                    code, label = c, text
                continue
            if _RULE.match(line) or _DEATHS.match(line):
                continue
            if m := _REC_START.match(line):
                close()
                builder = _FatalBuilder(doc, year, int(m.group(1)), m.group(2), m.group(3).strip())
                builder.rec.cause_code, builder.rec.cause_label = code, label
                builder.rec.parent_code, builder.rec.parent_label = parent_code, parent_label
                builder.raw.append(line)
                builder.raw_pages.append(page.page_number)
                continue
            if builder is not None:
                builder.feed(page.page_number, line)
    close()
    result.section_pages["fatal_accident_summaries"] = section_pages

    # The accident totals are written "stands at\n42(Including ...": catch wrapped ones too.
    flat = " ".join(" ".join(p.lines) for p in doc.pages if p.kind == "text")
    for m in _STATED_TOTAL.finditer(re.sub(r"\s+", " ", flat)):
        result.stated_fatal_totals.setdefault(int(m.group(1)), int(m.group(2)))


# --------------------------------------------------------------------------- major accidents
_DATE_TOKEN = re.compile(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$")
_TWO_INTS = re.compile(r"^(\d+)\s+(\d+)$")
_MAJOR_HEADER_TOKENS = {
    "sl.", "date of", "accident", "name of mine", "number of", "persons", "cause of accident",
    "killed", "s/injured", "killed s/injured", "10.", "major accident since 1901",
    "details of major accidents in coal mines (involving 4 or more deaths)",
    "during the year 1901-2022",
}


def _section_page_range(doc: PdfDocument, start: re.Pattern, end: re.Pattern) -> list[PageInfo]:
    pages, active = [], False
    for page in doc.pages:
        text = " ".join(" ".join(page.lines).split())
        if not active and start.search(text):
            active = True
        elif active and end.search(text):
            break
        if active:
            pages.append(page)
    return pages


def parse_major_accidents(doc: PdfDocument, result: ExtractionResult) -> None:
    pages = _section_page_range(
        doc,
        re.compile(r"MAJOR ACCIDENT SINCE 1901\s+Details of major accidents", re.I),
        re.compile(r"COURT OF INQUIRY\s+Particulars of Court", re.I),
    )
    result.section_pages["major_accidents_since_1901"] = [p.page_number for p in pages]
    if not pages:
        result.warnings.append("Section 'Major accident since 1901' was not found.")
        return

    tokens: list[tuple[int, str]] = []
    for page in pages:
        for line in page.lines:
            s = " ".join(line.split())
            if s and s.lower() not in _MAJOR_HEADER_TOKENS:
                tokens.append((page.page_number, s))

    starts: list[int] = []
    expected = 1
    for i in range(len(tokens) - 1):
        tok = tokens[i][1]
        if tok.isdigit() and _DATE_TOKEN.match(tokens[i + 1][1]):
            n = int(tok)
            if expected <= n <= expected + 3:
                if n != expected:
                    result.warnings.append(
                        f"Major-accident table: serial {expected} not found before serial {n} "
                        f"(page {tokens[i][0]})."
                    )
                starts.append(i)
                expected = n + 1

    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(tokens)
        page, serial = tokens[i][0], int(tokens[i][1])
        body = [t for _, t in tokens[i + 2 : end]]
        if k + 1 == len(starts):
            body = body[:8]  # do not swallow anything that trails the final row
        row = RawMajorRow(
            doc.filename, doc.category or "", serial, tokens[i + 1][1], "", None, None, "", page,
            " | ".join([tokens[i][1], tokens[i + 1][1], *body]),
        )
        split = None
        for j, tok in enumerate(body):
            if m := _TWO_INTS.match(tok):
                split = (j, j + 1, int(m.group(1)), int(m.group(2)))
                break
            if tok.isdigit() and j + 1 < len(body) and body[j + 1].isdigit() and j > 0:
                split = (j, j + 2, int(tok), int(body[j + 1]))
                break
        if split is None:
            row.mine_name = " ".join(body)
            row.warnings.append("Could not separate mine name, casualty counts and cause.")
        else:
            j, after, row.killed, row.seriously_injured = split
            row.mine_name = " ".join(body[:j]).strip()
            row.cause = " ".join(body[after:]).strip()
            if not row.mine_name:
                row.warnings.append("Mine name cell is empty.")
            if not row.cause:
                row.warnings.append("Cause cell is empty.")
        result.major_rows.append(row)


# --------------------------------------------------------------------------- courts of inquiry
_COURT_FIXED = re.compile(
    r"^\s*(?:(\d{1,3})\s+)?(\d{2}/\d{2}/\d{2,4})\s+(?:([a-z])\)\s*)?(.+?)(?:\s{2,}(\d+)(?:\s{2,}(.*))?)?\s*$"
)
_COURT_CAUSE = re.compile(r"^\s*\(([^)]+)\)\s*(.*)$")
_LEADING_INT = re.compile(r"^\s*(\d+)\s{2,}(.*)$")
_SUB_ITEM = re.compile(r"^([a-z])\)\s*(.+)$")


def parse_courts_of_inquiry(doc: PdfDocument, result: ExtractionResult) -> None:
    pages = _section_page_range(
        doc,
        re.compile(r"COURT OF INQUIRY\s+Particulars of Court", re.I),
        re.compile(r"MISCELLANEOUS\s+STATEMENT", re.I),
    )
    result.section_pages["courts_of_inquiry"] = [p.page_number for p in pages]
    if not pages:
        result.warnings.append("Section 'Court of Inquiry' was not found.")
        return
    fixed_hits = sum(
        1 for p in pages for line in p.lines
        if (m := _COURT_FIXED.match(line)) and m.group(5)
    )
    if fixed_hits >= 3:
        _parse_courts_fixed_width(doc, pages, result)
    else:
        _parse_courts_cell_per_line(doc, pages, result)


def _parse_courts_fixed_width(doc: PdfDocument, pages: list[PageInfo], result: ExtractionResult) -> None:
    current: RawCourtRow | None = None
    serial: int | None = None
    block: list[str] = []

    def close() -> None:
        nonlocal current, block
        if current is not None:
            current.source_text = "\n".join(block).rstrip()
            if current.cause is None:
                current.warnings.append("Cause (in brackets) not found for this row.")
            if current.killed is None:
                current.warnings.append("Number of persons killed not found for this row.")
            result.court_rows.append(current)
        current, block = None, []

    for page in pages:
        close()  # a row never continues onto the next page; do not absorb its table header
        for line in page.lines:
            if _RULE.match(line) or not line.strip():
                continue
            m = _COURT_FIXED.match(line)
            if m:
                close()
                if m.group(1):
                    serial = int(m.group(1))
                current = RawCourtRow(
                    doc.filename, doc.category or "", serial, m.group(3), m.group(2),
                    m.group(4).strip(), int(m.group(5)) if m.group(5) else None, None,
                    (m.group(6) or "").strip(), page.page_number, "",
                )
                block = [line.rstrip()]
                continue
            if current is None:
                continue
            block.append(line.rstrip())
            if current.cause is None and (c := _COURT_CAUSE.match(line)):
                current.cause = c.group(1).strip()
            elif current.killed is None and (k := _LEADING_INT.match(line)):
                current.killed = int(k.group(1))
                current.court_text = (current.court_text + " " + k.group(2)).strip()
    close()


def _parse_courts_cell_per_line(doc: PdfDocument, pages: list[PageInfo], result: ExtractionResult) -> None:
    tokens = [
        (p.page_number, " ".join(line.split())) for p in pages for line in p.lines if line.strip()
    ]
    starts = [i for i, (_, t) in enumerate(tokens) if _DATE_TOKEN.match(t)]
    serial: int | None = None
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(tokens)
        if i > 0 and tokens[i - 1][1].isdigit():
            serial = int(tokens[i - 1][1])
            if k + 1 < len(starts):
                pass
        body = [t for _, t in tokens[i + 1 : end]]
        # the next row's serial number is the last token of this row's body
        if k + 1 < len(starts) and body and body[-1].isdigit() and len(body) > 2:
            body = body[:-1]
        name, sub = (body[0] if body else ""), None
        if m := _SUB_ITEM.match(name):
            sub, name = m.group(1), m.group(2)
        killed = int(body[1]) if len(body) > 1 and body[1].isdigit() else None
        cause = next((c.group(1).strip() for t in body if (c := _COURT_CAUSE.match(t))), None)
        rest = [t for t in body[2:] if not _COURT_CAUSE.match(t)]
        row = RawCourtRow(
            doc.filename, doc.category or "", serial, sub, tokens[i][1], name.strip(), killed,
            cause, " ".join(rest), tokens[i][0], " | ".join([tokens[i][1], *body]),
        )
        row.warnings.append("Parsed from a table flattened to one cell per line; verify against the page.")
        if cause is None:
            row.warnings.append("Cause (in brackets) not found for this row.")
        result.court_rows.append(row)


# --------------------------------------------------------------------------- entry point
def extract_document(path: Path) -> ExtractionResult:
    """Read one PDF and run every parser. Raises PdfExtractionError if it is unreadable."""
    doc = read_pdf(path)
    result = ExtractionResult(document=doc)
    if doc.category is None:
        result.warnings.append(
            "Could not determine whether this volume covers coal or non-coal mines from its "
            "title pages; no records were extracted from it."
        )
        return result
    parse_fatal_summaries(doc, result)
    parse_major_accidents(doc, result)
    parse_courts_of_inquiry(doc, result)

    by_year: dict[int, list[int]] = {}
    for rec in result.fatal_records:
        by_year.setdefault(rec.year, []).append(rec.serial)
    for year, serials in sorted(by_year.items()):
        missing = sorted(set(range(1, max(serials) + 1)) - set(serials))
        dupes = sorted({s for s in serials if serials.count(s) > 1})
        if missing:
            result.warnings.append(f"{year}: serial numbers missing from the parsed summaries: {missing}")
        if dupes:
            result.warnings.append(f"{year}: serial numbers parsed more than once: {dupes}")
    log.info(
        "%s: %d fatal-accident records, %d major-accident rows, %d court-of-inquiry rows",
        doc.filename, len(result.fatal_records), len(result.major_rows), len(result.court_rows),
    )
    return result
