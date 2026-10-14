"""Citations: formatting, redaction, export, and verification against the PDF pages."""
from __future__ import annotations

from conftest import find_mine

from src.citations import citation_map, redact_person, render_evidence, verify_source_against_pdf
from src.report import plan_to_markdown
from src.schemas import SourceReference


def _source(pages, text="x") -> SourceReference:
    return SourceReference(source_id="S-1", source_filename="volume.pdf", page_start=min(pages), page_end=max(pages),
                           pages=pages, section="fatal_accident_summary", record_type="incident", source_text=text,
                           extraction_method="test", verification_status="parsed_clean")


def test_citation_format():
    assert _source([81]).citation() == "volume.pdf, p. 81"
    assert _source([118, 119]).citation() == "volume.pdf, pp. 118-119"


def test_victim_names_are_redacted_but_roles_kept():
    assert redact_person("1. P.Nonia, Trammer, Male, 57 Years") == "Trammer, Male, 57 Years"
    assert redact_person("unparseable") == "details not parsed"


def test_every_incident_and_finding_has_a_resolvable_citation(real_db):
    for mine in real_db.list_mines(water_only=True):
        incidents = real_db.incidents_for_mine(mine.mine_id)
        sources = real_db.get_sources(s for i in incidents for s in i.source_ids)
        for inc in incidents:
            assert inc.source_ids and all(s in sources for s in inc.source_ids)
            for s in inc.source_ids:
                assert sources[s].source_filename and sources[s].page_start >= 1 and sources[s].source_text.strip()
        for f in real_db.findings_for_mine(mine.mine_id):
            assert f.source_ids and all(s in sources for s in f.source_ids)


def test_cited_text_is_really_on_the_cited_pdf_pages(real_db, real_pdfs):
    by_name = {p.name: p for p in real_pdfs}
    checked = 0
    for mine in real_db.list_mines(water_only=True):
        for inc in real_db.incidents_for_mine(mine.mine_id, water_only=True):
            for src in real_db.get_sources(inc.source_ids).values():
                result = verify_source_against_pdf(src, by_name[src.source_filename])
                assert result["ratio"] >= 0.95, (src.source_id, result)
                checked += 1
    assert checked >= 70


def test_evidence_rendering_and_export_carry_citations(real_db, real_retriever, real_agent):
    mine = find_mine(real_db, "TURAMDIH URANIUM PROJECT", "metal_non_coal")
    bundle = real_retriever.build_evidence_bundle(mine.mine_id)
    cmap = citation_map(bundle)
    assert cmap["S-mnc-2016-F035"] == "DOC-20260831-WA0001..pdf, p. 81"
    text = render_evidence(bundle)
    assert 'source_ids="S-mnc-2016-F035"' in text and "212 cubic" in text and "geology / hydrogeology: NOT AVAILABLE" in text
    assert "Milan Karmakar" not in text and "GFM (Mining)" in text
    assert "Milan Karmakar" in render_evidence(bundle, redact_names=False)
    md = plan_to_markdown(real_agent.run(mine.mine_id).plan)
    assert "DOC-20260831-WA0001..pdf, p. 81" in md and "Decision-support aid only" in md
    assert "present status: unknown" in md and "## Missing information" in md
