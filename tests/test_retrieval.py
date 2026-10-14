"""Mine directory and historical retrieval, on the real extracted dataset."""
from __future__ import annotations

from conftest import find_mine

from src.ui import mine_label


# --------------------------------------------------------------------------- directory
def test_category_filters_return_only_that_category(real_db, real_retriever):
    counts = real_db.category_counts()
    assert set(counts) == {"coal", "metal_non_coal"}
    for category in counts:
        mines = real_db.list_mines(category)
        assert mines and all(m.mine_category == category for m in mines)
        assert len(mines) == counts[category]["mines"]
        water = real_db.list_mines(category, water_only=True)
        assert len(water) == counts[category]["water_mines"] and all(m.water_incident_count > 0 for m in water)
    coal_names = {m.mine_name for m in real_db.list_mines("coal")}
    assert "TURAMDIH URANIUM PROJECT" not in coal_names
    assert real_retriever.search_mines("turamdih", "coal").matches == []
    assert real_retriever.search_mines("turamdih", "metal_non_coal").matches[0].mine_name == "TURAMDIH URANIUM PROJECT"


def test_search_by_name_handles_case_and_spelling(real_retriever):
    exact = real_retriever.search_mines("chasnalla", "coal")
    assert {m.match_type for m in exact.matches[:2]} == {"exact"}
    assert real_retriever.search_mines("Kuardih", "coal").matches[0].mine_name == "KUARDIH"
    assert real_retriever.search_mines("chasnala", "coal").matches[0].mine_name.upper() == "CHASNALLA"
    assert real_retriever.search_mines("neyveli", "coal").matches[0].match_type == "prefix"


def test_search_by_location_and_structured_filters(real_retriever):
    by_text = real_retriever.search_mines("Hazaribagh", "coal")
    assert by_text.matches and all(m.district == "Hazaribagh" or "HAZARIBAGH" in m.mine_name.upper()
                                   for m in by_text.matches)
    by_filter = real_retriever.search_mines("", "coal", state="West Bengal", water_only=True)
    assert by_filter.matches and all(m.state == "West Bengal" and m.water_incident_count > 0 for m in by_filter.matches)
    narrowed = real_retriever.search_mines("", "coal", state="Jharkhand", district="Dhanbad")
    assert narrowed.matches and all(m.district == "Dhanbad" for m in narrowed.matches)


def test_similar_mine_names_are_distinguished(real_retriever):
    result = real_retriever.search_mines("chasnalla", "coal")
    assert result.ambiguous and len(result.matches) == 2
    assert len({m.mine_id for m in result.matches}) == 2
    labels = [mine_label(m) for m in result.matches]
    assert len(set(labels)) == 2 and any("Dhanbad" in x for x in labels) and any("not documented" in x for x in labels)
    neyveli = real_retriever.search_mines("neyveli no", "coal").matches
    assert {"NEYVELI NO 1", "NEYVELI NO 2"} <= {m.mine_name for m in neyveli}
    everything = real_retriever.search_mines("", None).matches
    assert len({m.mine_id for m in everything}) == len(everything)


def test_unknown_mine_gives_a_clear_empty_state(real_retriever):
    result = real_retriever.search_mines("zzqx nonexistent colliery", "coal")
    assert result.matches == [] and not result.ambiguous
    assert "No mine matching" in result.message and "does not mean" in result.message


# --------------------------------------------------------------------------- history
def test_only_records_of_the_selected_mine_are_returned(real_db, real_retriever):
    seen: dict[str, str] = {}
    for mine in real_db.list_mines():
        incidents = real_retriever.get_inundation_history(mine.mine_id, include_non_water=True)
        assert len(incidents) == mine.incident_count
        for inc in incidents:
            assert inc.mine_id == mine.mine_id
            assert seen.setdefault(inc.incident_id, mine.mine_id) == mine.mine_id
        water = real_retriever.get_inundation_history(mine.mine_id)
        assert len(water) == mine.water_incident_count
        assert all(f.mine_id == mine.mine_id for f in real_retriever.get_preventive_findings(mine.mine_id))
    assert len(seen) == real_db.stats()["incidents"]


def test_same_name_mines_do_not_share_records(real_db, real_retriever):
    located = find_mine(real_db, "CHASNALLA", "coal", located=True)
    historic = find_mine(real_db, "Chasnalla", "coal", located=False)
    a = {i.incident_id for i in real_retriever.get_inundation_history(located.mine_id, include_non_water=True)}
    b = {i.incident_id for i in real_retriever.get_inundation_history(historic.mine_id, include_non_water=True)}
    assert a and b and not (a & b)
    assert {i.date_text for i in real_retriever.get_inundation_history(historic.mine_id)} == {"27-12-1975", "05-04-1976"}
    assert historic.mine_id in located.possibly_related_mine_ids


def test_relevant_incidents_and_sources_are_retrieved(real_db, real_retriever):
    mine = find_mine(real_db, 'SAYAL "D"', "coal")
    incidents = real_retriever.get_inundation_history(mine.mine_id)
    assert [i.incident_id for i in incidents] == ["coal-2017-F054"]
    inc = incidents[0]
    assert inc.water_relevance == "irruption" and inc.incident_date == "2017-08-16" and inc.persons_killed_count == 2
    sources = real_db.get_sources(inc.source_ids)
    src = sources[inc.source_ids[0]]
    assert src.source_filename.lower().endswith(".pdf") and src.pages == [118, 119]
    assert "brick wall" in src.source_text and src.citation().endswith("pp. 118-119")
    findings = real_retriever.get_preventive_findings(mine.mine_id, [inc.incident_id])
    assert findings and all(f.incident_id == inc.incident_id and f.source_ids == [src.source_id] for f in findings)
    assert "retaining wall" in findings[0].measure_text and findings[0].current_status == "unknown"


def test_filters_by_year_and_type(real_db, real_retriever):
    mine = find_mine(real_db, "NEYVELI NO 2", "coal")
    all_water = real_retriever.get_inundation_history(mine.mine_id)
    assert len(all_water) == 3
    only_2018 = real_retriever.get_inundation_history(mine.mine_id, year_from=2018, year_to=2018)
    assert [i.incident_year for i in only_2018] == [2018]
    assert real_retriever.get_inundation_history(mine.mine_id, incident_types=["irruption"]) == []
    assert len(real_retriever.get_inundation_history(mine.mine_id, incident_types=["drowning"])) == 3


def test_empty_retrieval_creates_no_fictional_incidents(real_db, real_retriever):
    assert real_retriever.get_inundation_history("coal-does-not-exist-00000000") == []
    assert real_retriever.get_preventive_findings("coal-does-not-exist-00000000") == []
    assert real_retriever.get_mine_record("coal-does-not-exist-00000000") is None
    assert real_retriever.build_evidence_bundle("coal-does-not-exist-00000000") is None
    dry = next(m for m in real_db.list_mines("coal") if m.water_incident_count == 0)
    bundle = real_retriever.build_evidence_bundle(dry.mine_id)
    assert bundle.incidents == [] and bundle.findings == [] and not bundle.has_water_evidence


def test_hybrid_search_stays_inside_the_mine(real_db, real_retriever):
    mine = find_mine(real_db, "KUARDIH", "coal")
    own = {c["chunk_id"] for c in real_db.chunks_for_mine(mine.mine_id)}
    hits = real_retriever.hybrid_search(mine.mine_id, "inrush of water surface water jore withdrawal level", limit=10)
    assert hits and all(h.mine_id == mine.mine_id and h.chunk_id in own for h in hits)
    assert hits[0].incident_id == "coal-2017-F055" and hits[0].source is not None
    # a query that describes another mine's accident still cannot pull that mine's records in
    other = real_retriever.hybrid_search(mine.mine_id, "brick wall isolation stopping Sayal Hazaribagh", limit=10)
    assert all(h.mine_id == mine.mine_id for h in other)
    assert real_retriever.hybrid_search("coal-does-not-exist-00000000", "water") == []


def test_context_reports_missing_geology_and_separates_external_sources(real_db, real_retriever, tmp_path, monkeypatch):
    mine = find_mine(real_db, 'SAYAL "D"', "coal")
    ctx = real_retriever.get_context(mine.mine_id)
    assert all(i.kind == "in_corpus" for i in ctx.items)
    assert any("No geological" in x for x in ctx.limitations)
    # external context is accepted only with provenance and a retrieval date
    (tmp_path / "ok.json").write_text(
        '[{"mine_ids": ["%s"], "title": "T", "text": "Synthetic test note", "provenance": "Test source", '
        '"retrieved_on": "2026-01-15"}, {"mine_ids": "all", "title": "No provenance", "text": "x"}]' % mine.mine_id,
        encoding="utf-8")
    monkeypatch.setenv("SANKET_CONTEXT_DIR", str(tmp_path))
    from src.config import get_settings
    from src.retrievers import Retriever

    ctx2 = Retriever(real_db, None, get_settings()).get_context(mine.mine_id)
    external = [i for i in ctx2.items if i.kind == "external"]
    assert len(external) == 1 and external[0].provenance == "Test source" and external[0].retrieved_on == "2026-01-15"
    assert any("lacks" in x for x in ctx2.limitations)
