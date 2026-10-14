"""SANKET AI - home page and mine-category routing.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.schemas import CATEGORY_LABELS
from src.ui import (
    AUDIT_PAGE,
    DIRECTORY_PAGE,
    db_version,
    load_category_counts,
    load_report,
    page_setup,
    require_database,
)

page_setup("Home")

st.title("SANKET AI")
st.markdown('<p class="sk-sub">Historical Mine Inundation Intelligence &amp; Emergency Decision Support</p>',
            unsafe_allow_html=True)
st.write(
    "For officers taking charge of mines in an unfamiliar region. Select a mine to see the water-related "
    "accidents recorded for it in the DGMS SANKET volumes, what the DGMS inquiry found would have prevented "
    "them, and an evidence-grounded emergency response plan. Every finding links to the source page so it can "
    "be checked before action is taken."
)

db = require_database()
version = db_version()
counts = load_category_counts(version)
report = load_report(version) or {}

st.subheader("Select a mine category")
descriptions = {
    "coal": "Coal and lignite mines from the SANKET coal volume.",
    "metal_non_coal": "Metalliferous, stone, mineral and oil mines from the SANKET non-coal volume.",
}
columns = st.columns(2, gap="large")
for column, category in zip(columns, CATEGORY_LABELS):
    c = counts.get(category)
    with column:
        with st.container(border=True):
            st.markdown(f"### {CATEGORY_LABELS[category]}")
            st.caption(descriptions[category])
            if c is None:
                st.warning("No readable SANKET volume for this category was found during ingestion.")
            else:
                left, right = st.columns(2)
                left.metric("Mines with water-related records", c["water_mines"])
                right.metric("All mines in the records", c["mines"])
                st.caption(f"{c['water_incidents']} water-related records out of {c['incidents']} accident records.")
            if st.button(f"Open {CATEGORY_LABELS[category]} directory", key=f"open_{category}",
                         type="primary", width="stretch", disabled=c is None):
                st.session_state["category"] = category
                st.switch_page(DIRECTORY_PAGE)

st.divider()
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("What the records contain")
    st.markdown(
        "- **Fatal accidents, 2016 to 2022:** date, mine, owner, district, state, the DGMS narrative and the "
        "DGMS findings on what would have averted the accident.\n"
        "- **Major accidents since 1901** (four or more deaths) and **courts of inquiry:** date, mine name, "
        "casualties and cause only.\n"
        "- **Not in the records:** geology, water levels as measured values, measures implemented afterwards, "
        "non-fatal inundations at a named mine, and anything about present conditions."
    )
    st.caption("A mine with no record here has no *extracted* record. That is not evidence that it has never "
               "been inundated.")
with right:
    st.subheader("Source documents")
    for doc in db.documents():
        st.markdown(f"**{doc['title']}**")
        st.caption(f"{doc['filename']} | {doc['page_count']} pages | {doc['records']} records extracted")
    if report.get("generated_at"):
        st.caption(f"Extracted: {report['generated_at']}")
    st.page_link(AUDIT_PAGE, label="View the data extraction audit")
