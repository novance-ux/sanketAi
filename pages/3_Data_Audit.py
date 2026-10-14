"""Data extraction audit and system status."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.llm import data_flow_notice
from src.schemas import SECTION_LABELS
from src.ui import db_version, get_retriever, load_report, page_setup, require_database

page_setup("Data Audit")
db = require_database()
settings = get_settings()
report = load_report(db_version())

st.title("Data Audit and System Status")
if report is None:
    st.warning("The extraction report file is missing. Run `python -m src.ingestion --rebuild` to recreate it.")
    st.stop()

t = report["totals"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Raw records extracted", t["raw_records_extracted"])
c2.metric("Incidents", t["incidents"])
c3.metric("Water-related", t["water_related_incidents"])
c4.metric("Mines", t["mines"])
c5.metric("Preventive findings", t["preventive_findings"])
st.caption(f"Extracted {report['generated_at']} | pipeline version {report['pipeline_version']}")

st.subheader("Source documents")
for d in report["documents"]:
    with st.container(border=True):
        st.markdown(f"**{d['source_filename']}** - {d['volume']}")
        r = d["records_extracted"]
        st.markdown(
            f"- Pages processed: {d['pages_processed']} ({d['pages_with_selectable_text']} with selectable text; "
            f"{len(d['image_only_pages'])} image-only; {len(d['blank_pages'])} blank)\n"
            f"- Records: {r['fatal_accident_summaries']} fatal-accident summaries, "
            f"{r['major_accident_table_rows']} major-accident rows, {r['court_of_inquiry_rows']} court-of-inquiry rows\n"
            f"- OCR: {d['ocr']}"
        )
        st.dataframe(pd.DataFrame(d["fatal_accident_cross_check"]).rename(columns={
            "year": "Year", "stated_in_volume": "Fatal accidents stated in volume",
            "parsed": "Records parsed", "match": "Match"}), hide_index=True, width="stretch")
        for w in d["extraction_warnings"]:
            st.warning(w)
for f in report.get("unreadable_documents", []):
    st.error(f"{f['file']}: {f['error']}")

st.subheader("Field completeness")
st.dataframe(pd.DataFrame([{"Measure": k.replace("_", " "), "Count": v}
                           for k, v in report["field_completeness"].items()]), hide_index=True, width="stretch")
st.markdown("**Not present in the source volumes at all**")
for x in report["fields_absent_from_source"]:
    st.markdown(f"- {x}")

flagged = report["ambiguous_or_flagged_records"]
with st.expander(f"Ambiguous or flagged records ({len(flagged)})"):
    st.dataframe(pd.DataFrame([
        {"Source ID": a["source_id"], "File": a["source_filename"], "Pages": ", ".join(map(str, a["pages"])),
         "Section": SECTION_LABELS.get(a["section"], a["section"]), "Flag": " ".join(a["warnings"])}
        for a in flagged]), hide_index=True, width="stretch")
dupes = report["likely_duplicates_not_merged"]
with st.expander(f"Likely duplicate mines not merged ({len(dupes)}) - for human review"):
    st.caption("Look-alike names are never merged automatically. Confirmed variants go in config/mine_aliases.json.")
    st.dataframe(pd.DataFrame([
        {"Mine A": d["mine_a"]["name"], "Location A": f"{d['mine_a']['district'] or '-'}, {d['mine_a']['state'] or '-'}",
         "Mine B": d["mine_b"]["name"], "Location B": f"{d['mine_b']['district'] or '-'}, {d['mine_b']['state'] or '-'}",
         "Reason": d["reason"]} for d in dupes]), hide_index=True, width="stretch")
with st.expander(f"Records merged across sections ({len(report['record_merges'])}) and aliases applied "
                 f"({len(report['aliases_applied'])})"):
    st.dataframe(pd.DataFrame(report["record_merges"]), hide_index=True, width="stretch")
    if report["aliases_applied"]:
        st.dataframe(pd.DataFrame(report["aliases_applied"]), hide_index=True, width="stretch")

st.subheader("System status")
vec = report.get("vector_index", {})
st.markdown(
    f"- Structured database: `{settings.db_path}`\n"
    f"- Semantic index: {vec.get('status', 'unknown')}"
    + (f" ({vec.get('reason')})" if vec.get("reason") else "")
    + f"; {get_retriever().vectors.count() if get_retriever().vectors else 0} passages indexed\n"
    f"- Language model: {data_flow_notice()}\n"
    f"- Schema validation: {'no integrity problems' if not report['schema_validation_warnings'] else '; '.join(report['schema_validation_warnings'])}"
)
st.caption("To rebuild after the source documents change: `python -m src.ingestion --rebuild`.")
