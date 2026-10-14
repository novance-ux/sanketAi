"""Data ingestion: extraction from the real PDFs and normalisation logic on synthetic records."""
from __future__ import annotations

import dataclasses

import pytest

from src.config import get_settings
from src.data_normalizer import (
    CenturyTracker,
    classify_water,
    load_aliases,
    normalize,
    parse_source_date,
    split_findings,
)
from src.ingestion import IngestionError, build_chunks, run_ingestion
from src.pdf_extractor import PdfExtractionError, extract_document, read_pdf


# --------------------------------------------------------------------------- real PDFs
def test_both_pdfs_are_processed(real_extractions):
    assert set(real_extractions) == {"coal", "metal_non_coal"}
    for res in real_extractions.values():
        assert res.fatal_records and res.major_rows and res.court_rows
        assert res.document.page_count > 300


def test_parsed_fatal_counts_match_totals_stated_in_each_volume(real_extractions):
    for res in real_extractions.values():
        parsed = {}
        for rec in res.fatal_records:
            parsed[rec.year] = parsed.get(rec.year, 0) + 1
        assert res.stated_fatal_totals, "the volume's own yearly totals were not found"
        assert parsed == res.stated_fatal_totals
        assert not res.warnings


def test_page_references_are_preserved_and_valid(real_extractions):
    for res in real_extractions.values():
        last = res.document.page_count
        for rec in res.fatal_records:
            assert rec.pages and all(1 <= p <= last for p in rec.pages)
            assert rec.pages == sorted(rec.pages) and rec.pages[-1] - rec.pages[0] <= 3
            assert rec.source_text and rec.mine_name
        assert all(1 <= r.page <= last for r in res.major_rows + res.court_rows)


def test_table_rows_keep_their_cells_together(real_extractions):
    coal = real_extractions["coal"]
    serials = [r.serial for r in coal.major_rows]
    assert serials == list(range(1, len(serials) + 1))
    row = next(r for r in coal.major_rows if r.mine_name == "Chasnalla" and r.date_text == "27-12-1975")
    assert (row.killed, row.seriously_injured, row.cause, row.page) == (375, 0, "Irruption of Water", 304)
    court = next(r for r in coal.court_rows if r.mine_name == "Chasnalla" and r.date_text == "27/12/75")
    assert (court.killed, court.cause) == (375, "Inundation")
    multi = next(r for r in real_extractions["metal_non_coal"].major_rows if r.serial == 100)
    assert multi.mine_name == "Sindesar Khurd Galena & Sphalarite" and multi.killed == 4


def test_known_record_fields_are_extracted_verbatim(real_extractions):
    rec = next(r for r in real_extractions["coal"].fatal_records if r.year == 2017 and r.serial == 54)
    assert rec.mine_name == 'SAYAL "D"' and rec.cause_code == "0991"
    assert (rec.district, rec.state, rec.owner) == ("Hazaribagh", "Jharkhand", "Central Coalfields Ltd.")
    assert rec.pages == [118, 119] and len(rec.persons_killed) == 2
    assert rec.narrative.startswith("While a brick wall") and "washed away" in rec.narrative
    assert rec.findings_text.startswith("Had") and "Regulation 129(1)" in rec.findings_text


def test_missing_pdf_is_reported(tmp_path):
    with pytest.raises(PdfExtractionError, match="not found"):
        read_pdf(tmp_path / "absent.pdf")


def test_corrupted_pdf_is_reported(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.5 this is not a real pdf")
    with pytest.raises(PdfExtractionError):
        extract_document(bad)


def test_ingestion_stops_without_pdfs_and_never_substitutes_sample_data(tmp_path, monkeypatch):
    monkeypatch.setenv("SANKET_PDF_DIRS", str(tmp_path / "empty"))
    monkeypatch.setenv("SANKET_PROCESSED_DIR", str(tmp_path / "out"))
    with pytest.raises(IngestionError, match="No PDF files"):
        run_ingestion(get_settings(), skip_vectors=True)
    assert not (tmp_path / "out" / "sanket.sqlite3").exists()


def test_ingestion_with_only_unreadable_pdfs_is_a_blocker(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "junk.pdf").write_bytes(b"not a pdf at all")
    monkeypatch.setenv("SANKET_PDF_DIRS", str(tmp_path / "raw"))
    monkeypatch.setenv("SANKET_PROCESSED_DIR", str(tmp_path / "out"))
    with pytest.raises(IngestionError, match="junk.pdf"):
        run_ingestion(get_settings(), skip_vectors=True)


def test_ingestion_rerun_is_incremental_and_writes_a_report(real_pdfs, tmp_path, monkeypatch):
    monkeypatch.setenv("SANKET_PDF_DIRS", ";".join(sorted({str(p.parent) for p in real_pdfs})))
    monkeypatch.setenv("SANKET_PROCESSED_DIR", str(tmp_path / "out"))
    settings = get_settings()
    before = {p: p.stat().st_mtime_ns for p in real_pdfs}
    first = run_ingestion(settings, skip_vectors=True)
    assert not first.get("skipped") and settings.report_json.is_file() and settings.report_md.is_file()
    assert all(d["pages_processed"] > 300 for d in first["documents"])
    assert first["totals"]["raw_records_extracted"] == sum(d["records_extracted"]["total"] for d in first["documents"])
    second = run_ingestion(settings, skip_vectors=True)
    assert second.get("skipped") is True
    assert second["totals"] == first["totals"]
    assert {p: p.stat().st_mtime_ns for p in real_pdfs} == before, "source PDFs must never be modified"


# --------------------------------------------------------------------------- normalisation logic
def test_dates_are_parsed_without_guessing():
    assert parse_source_date("16.08.17", two_digit_base=2000)[:3] == ("2017-08-16", 2017, "exact")
    assert parse_source_date("27-12-1975")[:3] == ("1975-12-27", 1975, "exact")
    tracker = CenturyTracker()
    assert parse_source_date("26/09/04", tracker=tracker)[:3] == ("1904-09-26", 1904, "century_inferred")
    assert parse_source_date("28/08/94", tracker=tracker)[1] == 1994
    assert parse_source_date("31/01/17", tracker=tracker)[1] == 2017
    # hyphenated two-digit date with day and month both <= 12: no ISO date is invented
    iso, year, quality, _ = parse_source_date("05-11-14", tracker=CenturyTracker())
    assert (iso, year, quality) == (None, 1914, "day_month_ambiguous")
    assert parse_source_date("sometime", two_digit_base=2000)[:3] == (None, None, "unparsed")


def test_water_classification_states_its_basis():
    assert classify_water("0991", "Irruption of Water", "x")[0] == "irruption"
    assert classify_water(None, "Inundation", None)[0] == "irruption"
    assert classify_water("0993", "Drowning in Water", "x")[0] == "drowning"
    kind, basis = classify_water("0335", "Dumpers", "the dumper fell into the sump and the operator drowned")
    assert kind == "water_keyword" and "Dumpers" in basis and "drowned" in basis
    assert classify_water("0335", "Dumpers", "a water tanker was reversing") == ("none", None)


def test_findings_are_split_and_keep_source_wording():
    items, quality = split_findings(
        "Had,\n\ni) the dam been of standard design\nas required,\n\nii) persons not been deployed,\n\n"
        "this accident could have been averted.")
    assert items == ["the dam been of standard design as required", "persons not been deployed"]
    assert quality == "itemised"
    single, quality = split_findings("Had, the sump been fenced, this accident could have been averted.")
    assert single == ["the sump been fenced"] and quality == "unsplit_block"
    assert split_findings("") == ([], "itemised")


def test_duplicates_and_similar_names_are_handled_safely(synthetic_dataset):
    ds = synthetic_dataset
    coal = [m for m in ds.mines if m.mine_category == "coal"]
    names = sorted(m.mine_name for m in coal)
    # same name + same location -> one mine with two incidents
    alpha = [m for m in coal if m.mine_name == "TESTMINE ALPHA" and m.state == "Teststate"]
    assert len(alpha) == 1 and alpha[0].incident_count == 2 and alpha[0].water_incident_count == 1
    # same name, different location -> separate mines, flagged as possibly related
    assert sum(m.mine_name == "TESTMINE ALPHA" for m in coal) == 2
    assert alpha[0].possibly_related_mine_ids
    # look-alike names are never merged automatically
    assert "TESTMINE BETA OC-II" in names and "TESTMINE BETA OC-III" in names
    assert "Testmine Delta" in names and "Testmine Dellta" in names
    reported = {(d["mine_a"]["name"], d["mine_b"]["name"]) for d in ds.duplicate_candidates}
    assert ("Testmine Delta", "Testmine Dellta") in reported
    # the same category boundary is respected: the non-coal mine of the same name is its own mine
    assert any(m.mine_name == "TESTMINE ALPHA" and m.mine_category == "metal_non_coal" for m in ds.mines)
    assert not ds.warnings
    assert len({i.incident_id for i in ds.incidents}) == len(ds.incidents)


def test_table_and_court_rows_for_the_same_accident_become_one_incident(synthetic_dataset):
    gamma = [i for i in synthetic_dataset.incidents if i.mine_name_in_source == "Testmine Gamma"]
    assert len(gamma) == 1
    assert gamma[0].record_sections == ["major_accident_table", "court_of_inquiry_table"]
    assert len(gamma[0].source_ids) == 2 and "Inundation" in gamma[0].court_of_inquiry


def test_curated_alias_merges_spelling_variants(synthetic_results, tmp_path):
    alias_file = tmp_path / "aliases.json"
    alias_file.write_text('{"_comment": "x", "coal": {"Testmine Dellta": "Testmine Delta"}}', encoding="utf-8")
    ds = normalize(synthetic_results, load_aliases(alias_file))
    delta = [m for m in ds.mines if m.mine_name in ("Testmine Delta", "Testmine Dellta")]
    assert len(delta) == 1 and delta[0].incident_count == 2 and delta[0].name_variants
    assert ds.aliases_applied


def test_nulls_and_ambiguous_values_are_not_fabricated(synthetic_dataset):
    ds = synthetic_dataset
    table = next(i for i in ds.incidents if i.mine_name_in_source == "Testmine Delta")
    assert table.description is None and table.findings_text is None and table.time_text is None
    mica = next(i for i in ds.incidents if i.mine_name_in_source == "Testmine Mica")
    assert mica.incident_date is None and mica.incident_year == 1914 and mica.date_text == "05-11-14"
    name_only = next(m for m in ds.mines if m.mine_name == "Testmine Delta")
    assert (name_only.state, name_only.district, name_only.operator) == (None, None, None)
    assert name_only.identity_basis == "name_only"
    assert all(m.geology_summary is None for m in ds.mines)
    assert all(f.current_status == "unknown" and f.implementation_date is None and f.reported_outcome is None
               for f in ds.findings)


def test_chunks_keep_findings_attached_to_their_mine_and_incident(synthetic_dataset):
    chunks = build_chunks(synthetic_dataset)
    by_id = {i.incident_id: i for i in synthetic_dataset.incidents}
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    for c in chunks:
        inc = by_id[c["incident_id"]]
        assert c["mine_id"] == inc.mine_id and c["source_id"] in inc.source_ids
        assert inc.mine_name_in_source in c["text"]
    assert any(c["chunk_type"] == "findings" and "standard design" in c["text"] for c in chunks)


def test_year_mismatch_is_flagged_not_silently_corrected(synthetic_results):
    rec = synthetic_results[0].fatal_records[0]
    synthetic_results[0].fatal_records[0] = dataclasses.replace(rec, date_text="16.08.15")
    ds = normalize(synthetic_results, {})
    inc = next(i for i in ds.incidents if i.incident_id == "coal-2017-F001")
    assert inc.incident_date is None and inc.incident_year == 2017 and inc.date_text == "16.08.15"
    src = next(s for s in ds.sources if s.source_id == inc.source_ids[0])
    assert src.verification_status == "needs_review"
