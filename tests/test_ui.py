"""UI behaviour, driven headlessly with Streamlit's AppTest against the real dataset."""
from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from conftest import find_mine
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
HOME = str(ROOT / "app.py")
DIRECTORY = str(ROOT / "pages" / "1_Mine_Directory.py")
DASHBOARD = str(ROOT / "pages" / "2_Mine_Intelligence.py")
AUDIT = str(ROOT / "pages" / "3_Data_Audit.py")


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    monkeypatch.chdir(ROOT)
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def _run(path: str, **state) -> AppTest:
    # Always enter through app.py, as a user does, so page links resolve as in the real app.
    at = AppTest.from_file(HOME, default_timeout=120)
    for key, value in state.items():
        at.session_state[key] = value
    if path != HOME:
        at.switch_page(str(Path(path).relative_to(ROOT)).replace("\\", "/"))
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _all_text(at: AppTest) -> str:
    parts = []
    for kind in ("markdown", "caption", "info", "warning", "error", "success", "title", "subheader", "text"):
        parts += [str(e.value) for e in getattr(at, kind)]
    return "\n".join(parts)


def test_home_shows_counts_computed_from_the_records(real_db):
    at = _run(HOME)
    counts = real_db.category_counts()
    assert at.title[0].value == "SANKET AI"
    shown = {m.label: m.value for m in at.metric}
    assert str(counts["coal"]["water_mines"]) in [m.value for m in at.metric]
    assert [m.value for m in at.metric] == [
        str(counts["coal"]["water_mines"]), str(counts["coal"]["mines"]),
        str(counts["metal_non_coal"]["water_mines"]), str(counts["metal_non_coal"]["mines"])]
    assert "Mines with water-related records" in shown
    labels = [b.label for b in at.button if b.key != "sanket_fab_btn"]
    assert labels == ["Open Coal Mines directory", "Open Metal / Non-Coal Mines directory"]
    assert at.button(key="sanket_fab_btn").label == "Ask SANKET"  # floating chatbot on every page


def test_category_selection_sets_the_category(real_db):
    at = _run(HOME)
    at.button(key="open_metal_non_coal").click().run()
    assert at.session_state["category"] == "metal_non_coal"


def test_directory_lists_only_the_selected_category(real_db):
    at = _run(DIRECTORY, category="metal_non_coal")
    table = at.dataframe[0].value
    expected = {m.mine_name for m in real_db.list_mines("metal_non_coal", water_only=True)}
    assert set(table["Mine"]) == expected and "TURAMDIH URANIUM PROJECT" in expected
    at.radio[0].set_value("coal").run()
    assert not at.exception
    coal = set(at.dataframe[0].value["Mine"])
    assert "KUARDIH" in coal and "TURAMDIH URANIUM PROJECT" not in coal


def test_directory_search_and_empty_state(real_db):
    at = _run(DIRECTORY, category="coal")
    at.text_input[0].set_value("chasnalla").run()
    table = at.dataframe[0].value
    assert list(table["Mine"].str.upper()) == ["CHASNALLA", "CHASNALLA"]
    assert set(table["District"]) == {"Dhanbad", "-"}
    at.text_input[0].set_value("zzqx nonexistent colliery").run()
    assert not at.exception and len(at.dataframe) == 0
    assert any("No mine matching" in i.value for i in at.info)


def test_directory_selection_opens_the_dashboard_for_that_mine(real_db):
    mine = find_mine(real_db, "KUARDIH", "coal")
    at = _run(DIRECTORY, category="coal")
    at.selectbox(key="dir_pick_coal").set_value(mine.mine_id).run()
    next(b for b in at.button if b.label == "Open dashboard").click().run()
    assert at.session_state["mine_id"] == mine.mine_id


def test_dashboard_without_a_selection_explains_what_to_do(real_db):
    at = _run(DASHBOARD)
    assert any("No mine is selected" in i.value for i in at.info)


def test_dashboard_updates_with_the_selected_mine(real_db):
    kuardih = find_mine(real_db, "KUARDIH", "coal")
    at = _run(DASHBOARD, mine_id=kuardih.mine_id)
    assert at.title[0].value == "KUARDIH"
    text = _all_text(at)
    assert "jore water" in text and "sanket0404_2024.pdf, p. 119" in text
    assert "brick wall" not in text  # another mine's narrative must not appear
    sayal = find_mine(real_db, 'SAYAL "D"', "coal")
    at2 = _run(DASHBOARD, mine_id=sayal.mine_id)
    text2 = _all_text(at2)
    assert at2.title[0].value == 'SAYAL "D"' and "brick wall" in text2 and "jore water" not in text2


def test_dashboard_renders_incidents_findings_and_sources(real_db):
    mine = find_mine(real_db, 'SAYAL "D"', "coal")
    at = _run(DASHBOARD, mine_id=mine.mine_id)
    text = _all_text(at)
    assert "Documented in source" in text and "Not documented in source" in text
    assert "retaining wall" in text and "Present status: unknown" in text
    assert "NOT in place at the time" in text
    expanders = [e.label for e in at.expander]
    assert any("16.08.17" in x and "Irruption of Water" in x for x in expanders)
    assert any(x.startswith("Source: sanket0404_2024.pdf, pp. 118-119") for x in expanders)
    assert [m.value for m in at.metric][:1] == ["1"]


def test_dashboard_generates_a_labelled_plan(real_db):
    mine = find_mine(real_db, 'SAYAL "D"', "coal")
    at = _run(DASHBOARD, mine_id=mine.mine_id)
    assert any("No plan has been generated" in i.value for i in at.info)
    next(b for b in at.button if b.label == "Generate plan").click().run()
    assert not at.exception
    text = _all_text(at)
    assert "Decision-support aid only" in text
    for title in ("Initial Detection and Verification", "Recovery and Prevention"):
        assert title in text
    assert "Based on this mine" in text and "General practice - not from this mine" in text
    assert "Offline rule engine" in text and "No geological" in text
    assert mine.mine_id in at.session_state["plans"]


def test_dashboard_empty_state_for_a_mine_without_water_records(real_db):
    dry = next(m for m in real_db.list_mines("coal") if m.water_incident_count == 0 and m.state)
    at = _run(DASHBOARD, mine_id=dry.mine_id)
    text = _all_text(at)
    assert "No water-related record for this mine was found" in text
    assert "never experienced inundation" in text
    next(b for b in at.button if b.label == "Generate plan").click().run()
    assert any("GENERIC SAFETY FRAMEWORK" in w.value for w in at.warning)


def test_name_only_mine_shows_identity_note_and_missing_fields(real_db):
    mine = find_mine(real_db, "Chasnalla", "coal", located=False)
    at = _run(DASHBOARD, mine_id=mine.mine_id)
    text = _all_text(at)
    assert "Identity note" in text and "No narrative in source" in text
    assert "Possibly related records" in text


def test_stale_mine_id_is_handled(real_db):
    at = _run(DASHBOARD, mine_id="coal-removed-mine-00000000")
    assert any("no longer in the dataset" in i.value for i in at.info)


def test_missing_database_shows_recovery_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("SANKET_PROCESSED_DIR", str(tmp_path / "nothing"))
    for page in (HOME, DIRECTORY, DASHBOARD, AUDIT):
        st.cache_resource.clear()
        at = AppTest.from_file(HOME, default_timeout=60)
        if page != HOME:
            at.switch_page(str(Path(page).relative_to(ROOT)).replace("\\", "/"))
        at.run()
        assert not at.exception
        assert any("has not been built" in e.value for e in at.error)
        assert any("python -m src.ingestion" in m.value for m in at.markdown)


def test_audit_page_renders(real_db):
    at = _run(AUDIT)
    assert at.title[0].value == "Data Audit and System Status"
    assert len(at.dataframe) >= 3 and int(at.metric[0].value) > 1000
