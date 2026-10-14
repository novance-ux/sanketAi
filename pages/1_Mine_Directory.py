"""Screen B - searchable mine directory for the selected category."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.schemas import CATEGORY_LABELS
from src.ui import (
    db_version,
    get_retriever,
    load_locations,
    mine_label,
    open_mine,
    page_setup,
    place_of,
    require_database,
)

page_setup("Mine Directory")
require_database()
version = db_version()

categories = list(CATEGORY_LABELS)
if st.session_state.get("category") not in categories:
    st.session_state["category"] = categories[0]

st.title("Mine Directory")
category = st.radio("Mine category", categories, format_func=CATEGORY_LABELS.get, horizontal=True, key="category")

locations = load_locations(version, category)
c1, c2, c3 = st.columns([3, 2, 2])
query = c1.text_input("Search by mine name, district, state or owner", key="dir_query",
                      placeholder="e.g. Chasnalla, Dhanbad, Singareni")
state = c2.selectbox("State", ["All states", *locations], key=f"dir_state_{category}")
state = None if state == "All states" else state
district = c3.selectbox("District", ["All districts", *(locations.get(state, []) if state else [])],
                        key=f"dir_district_{category}_{state}", disabled=state is None)
district = None if district == "All districts" else district
water_only = st.toggle("Show only mines with water-related records", value=True, key="dir_water_only",
                       help="Turn off to list every mine that appears in the accident records.")
if state:
    st.caption("State and district are documented only for mines in the 2016-2022 fatal-accident summaries. "
               "Mines that appear only in the historic tables have no location in the source and are hidden "
               "by a location filter.")

result = get_retriever().search_mines(query, category, state=state, district=district, water_only=water_only)

if not result.matches:
    st.info(result.message)
    if water_only:
        st.caption("Try turning off the water-related filter, or check the spelling. Mines are listed exactly "
                   "as printed in the SANKET volumes.")
    st.stop()

st.caption(result.message + (" Similar names are distinguished by district, state and owner."
                             if result.ambiguous else ""))

by_id = {m.mine_id: m for m in result.matches}
labels = {m.mine_id: mine_label(m) for m in result.matches}
left, right = st.columns([5, 1], vertical_alignment="bottom")
chosen = left.selectbox("Select a mine (type to filter)", list(by_id), format_func=labels.get, index=None,
                        placeholder="Choose a mine to open its dashboard", key=f"dir_pick_{category}")
if right.button("Open dashboard", type="primary", disabled=chosen is None, width="stretch"):
    open_mine(chosen)

table = pd.DataFrame([
    {
        "Mine": m.mine_name,
        "District": m.district or "-",
        "State": m.state or "-",
        "Owner (as printed)": m.operator or "-",
        "Water-related records": m.water_incident_count,
        "All accident records": m.incident_count,
        "Match": m.match_type if query else "",
        "mine_id": m.mine_id,
    }
    for m in result.matches
])
st.caption("Select a row to open that mine. '-' means the source does not state the value.")
event = st.dataframe(
    table.drop(columns=["mine_id"] + ([] if query else ["Match"])),
    hide_index=True, width="stretch", height=min(560, 38 * (len(table) + 1) + 4),
    on_select="rerun", selection_mode="single-row", key=f"dir_table_{category}_{water_only}",
)
rows = event.selection.rows if event and event.selection else []
if rows:
    picked = by_id[table.iloc[rows[0]]["mine_id"]]
    st.success(f"Selected: {picked.mine_name} ({place_of(picked)})")
    if st.button(f"Open dashboard for {picked.mine_name}", type="primary", key="dir_open_row"):
        open_mine(picked.mine_id)
