"""Screen C - mine-specific intelligence dashboard."""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agent import MineNotFound
from src.llm import data_flow_notice, is_llm_enabled
from src.report import plan_to_markdown
from src.schemas import CATEGORY_LABELS, SECTION_LABELS, WATER_LABELS, Incident, SourceReference
from src.ui import (
    DIRECTORY_PAGE,
    NOT_DOCUMENTED,
    SERIES_STYLE,
    badge,
    banner,
    get_agent,
    get_retriever,
    md_escape,
    open_mine,
    page_setup,
    place_of,
    quote_block,
    require_database,
)

page_setup("Mine Intelligence")
db = require_database()
retriever = get_retriever()

mine_id = st.session_state.get("mine_id")
mine = db.get_mine(mine_id) if mine_id else None
if mine is None:
    st.title("Mine Intelligence Dashboard")
    st.info("No mine is selected." if not mine_id else
            "The selected mine is no longer in the dataset (the records were rebuilt). Please select it again.")
    st.page_link(DIRECTORY_PAGE, label="Go to the mine directory")
    st.stop()

bundle = retriever.build_evidence_bundle(mine.mine_id)
sources = bundle.sources


def source_panel(source_ids: list[str], key: str) -> None:
    """Expandable supporting source details: file, pages, section and verbatim text."""
    for sid in source_ids:
        src: SourceReference | None = sources.get(sid)
        if src is None:
            continue
        with st.expander(f"Source: {src.citation()} - {SECTION_LABELS.get(src.section, src.section)}"):
            st.caption(f"Source ID `{src.source_id}` | extraction: {src.extraction_method} | "
                       f"status: {src.verification_status.replace('_', ' ')}")
            for w in src.warnings:
                st.warning(w)
            st.text(src.source_text)


def field(label: str, value) -> None:
    if value in (None, "", []):
        st.markdown(f"**{label}:** " + badge("missing", NOT_DOCUMENTED), unsafe_allow_html=True)
    else:
        st.markdown(f"**{label}:** {md_escape(str(value))}")


# --------------------------------------------------------------------------- header
st.caption(f"{CATEGORY_LABELS[mine.mine_category]}  /  Mine directory  /  Dashboard")
st.title(mine.mine_name)
st.markdown(
    f"{CATEGORY_LABELS[mine.mine_category]} | {md_escape(place_of(mine))} | "
    f"Owner as printed: {md_escape(mine.operator) if mine.operator else NOT_DOCUMENTED}"
)
if mine.identity_basis == "name_only":
    banner("Identity note: the source lists this mine by name only (historic tables). State, district and owner "
           "are not printed, so confirm that this is the mine you intend before relying on these records.")

tab_overview, tab_timeline, tab_measures, tab_plan, tab_search = st.tabs([
    "A. Overview", "B. Inundation timeline", "C. Preventive findings",
    "D. Emergency response plan", "Search this mine's records",
])

# --------------------------------------------------------------------------- A. overview
with tab_overview:
    water, other = bundle.incidents, bundle.other_incidents
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Water-related records", len(water))
    c2.metric("Earliest water-related record", min((i.incident_year for i in water if i.incident_year), default="-"))
    c3.metric("Latest water-related record", max((i.incident_year for i in water if i.incident_year), default="-"))
    c4.metric("Other fatal-accident records", len(other))

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Mine record")
        st.markdown(badge("doc", "Documented in source"), unsafe_allow_html=True)
        field("Mine name", mine.mine_name)
        if mine.name_variants:
            field("Other spellings in the source", "; ".join(mine.name_variants))
        field("Category", CATEGORY_LABELS[mine.mine_category])
        field("State", mine.state)
        field("District", mine.district)
        field("DGMS region", None)
        field("Owner / operator", mine.operator)
        if mine.operator_variants:
            field("Owner printed differently in other records", "; ".join(mine.operator_variants))
        field("Geology / hydrogeology", mine.geology_summary)
        st.caption(f"Mine ID: {mine.mine_id}")

        st.subheader("Documented risk indicators")
        if not water:
            st.info("No water-related record for this mine was found in the processed documents. "
                    "This does not establish that the mine has never experienced inundation.")
        else:
            st.markdown(
                f"- Irruption of water / inundation (DGMS classification): **{mine.irruption_count}**\n"
                f"- Drowning in water (DGMS classification): **{mine.drowning_count}**\n"
                f"- Water mentioned in the narrative of an accident classified otherwise: **{mine.water_keyword_count}**\n"
                f"- Persons killed in these records: **{sum(i.persons_killed_count or 0 for i in water)}**\n"
                f"- DGMS findings on precautions not in place: **{len(bundle.findings)}**"
            )
            st.caption("These are counts of documented records, not a risk score or a forecast.")
    with right:
        st.subheader("Dataset coverage and completeness")
        sections = sorted({s for i in [*water, *other] for s in i.record_sections})
        st.markdown("Records for this mine come from: " + "; ".join(SECTION_LABELS[s] for s in sections) + ".")
        for gap in bundle.data_gaps:
            st.markdown(f"- {md_escape(gap)}")
        st.caption("Current status, active water levels and present-day safety conditions are not known to this system.")

    if mine.possibly_related_mine_ids:
        st.subheader("Possibly related records (same name, identity not verified)")
        st.caption("These records carry the same mine name, but the source prints no location for at least one "
                   "side, so they are kept separate and are not used in this mine's plan.")
        for rid in mine.possibly_related_mine_ids:
            rel = db.get_mine(rid)
            if rel is None:
                continue
            a, b = st.columns([5, 1], vertical_alignment="center")
            a.markdown(f"**{md_escape(rel.mine_name)}** - {md_escape(place_of(rel))} | "
                       f"{rel.water_incident_count} water-related of {rel.incident_count} records "
                       f"({rel.earliest_year}-{rel.latest_year})")
            if b.button("Open", key=f"rel_{rid}"):
                open_mine(rid)

    if other:
        with st.expander(f"Other fatal accidents recorded for this mine ({len(other)}) - not water-related"):
            st.dataframe(pd.DataFrame([
                {"Date (as printed)": i.date_text, "Year": i.incident_year, "Cause in source": i.cause_label,
                 "Killed": i.persons_killed_count,
                 "Source": "; ".join(sources[s].citation() for s in i.source_ids if s in sources)}
                for i in other]), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- B. timeline
with tab_timeline:
    st.subheader("Historical inundation timeline")
    show_other = st.toggle("Also show accidents that are not water-related", value=False, key="tl_other")
    shown: list[Incident] = sorted(
        [*bundle.incidents, *(bundle.other_incidents if show_other else [])],
        key=lambda i: (i.incident_year or 0, i.incident_date or "", i.incident_id))
    if not bundle.incidents:
        st.info("No matching water-related record was found in the processed documents for this mine. "
                "This is not a finding that the mine has never experienced inundation: the yearly narratives "
                "cover fatal accidents from 2016 to 2022 only, and the historic tables list only accidents "
                "with four or more deaths.")
    if shown:
        fig = go.Figure()
        for key, style in SERIES_STYLE.items():
            items = [i for i in shown if i.water_relevance == key and i.incident_year]
            if not items:
                continue
            fig.add_trace(go.Scatter(
                x=[i.incident_date or f"{i.incident_year}-07-01" for i in items],
                y=[style["label"]] * len(items), mode="markers", name=style["label"],
                marker=dict(color=style["color"], symbol=style["symbol"], opacity=0.9,
                            size=[12 + 5 * math.sqrt(min(i.persons_killed_count or 1, 400)) for i in items],
                            line=dict(width=2, color="#fcfcfb")),
                customdata=[[i.date_text, i.cause_label or "not stated",
                             i.persons_killed_count if i.persons_killed_count is not None else "not stated",
                             "; ".join(sources[s].citation() for s in i.source_ids if s in sources)] for i in items],
                hovertemplate="<b>%{customdata[0]}</b><br>Cause in source: %{customdata[1]}"
                              "<br>Persons killed: %{customdata[2]}<br>%{customdata[3]}<extra></extra>",
            ))
        fig.update_layout(
            height=150 + 46 * len(fig.data), margin=dict(l=10, r=10, t=36, b=10),
            paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", font=dict(color="#52514e", size=13),
            legend=dict(orientation="h", y=1.18, x=0), hoverlabel=dict(bgcolor="#ffffff"),
            xaxis=dict(type="date", gridcolor="#ecebe6", linecolor="#c3c2b7", title=None),
            yaxis=dict(gridcolor="#ecebe6", title=None, categoryorder="array",
                       categoryarray=[s["label"] for s in reversed(list(SERIES_STYLE.values()))]),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Marker area grows with the number of persons killed. Records with only a year are placed "
                   "mid-year. The same records are listed below with their sources.")

    for inc in shown:
        style = SERIES_STYLE[inc.water_relevance]
        killed = f"{inc.persons_killed_count} killed" if inc.persons_killed_count is not None else "casualties not stated"
        with st.expander(f"{inc.date_text}  |  {inc.cause_label or 'cause not stated'}  |  {killed}  |  {style['label']}",
                         expanded=len(shown) <= 2 and inc.water_relevance != "none"):
            st.markdown(badge("doc", "Documented in source")
                        + f"<span class='sk-muted'>{'; '.join(sources[s].citation() for s in inc.source_ids if s in sources)}</span>",
                        unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                field("Date as printed", inc.date_text)
                if inc.date_quality != "exact":
                    st.caption({"century_inferred": "Century inferred from the table order.",
                                "day_month_ambiguous": "Day and month cannot be told apart; only the year is reliable.",
                                "unparsed": "The printed date could not be parsed."}[inc.date_quality])
                field("Time as printed", inc.time_text)
                field("Cause as classified by DGMS",
                      f"{inc.cause_label} (code {inc.cause_code})" if inc.cause_code else inc.cause_label)
                field("Why this record is shown as water-related", inc.water_basis)
            with b:
                field("Persons killed", inc.persons_killed_count)
                field("Seriously injured", inc.seriously_injured_count)
                field("Water level / volume / rate (as a recorded field)", None)
                field("Quantities mentioned in the narrative", "; ".join(inc.measurements_mentioned))
            if inc.persons_killed:
                field("Casualties (as printed)", "; ".join(inc.persons_killed))
            st.markdown("**Incident description, mechanism and affected workings (DGMS narrative, verbatim):**")
            if inc.description:
                quote_block(inc.description)
            else:
                st.markdown(badge("missing", "No narrative in source - table entry only"), unsafe_allow_html=True)
            st.markdown("**Reported cause / contributing factors (DGMS findings, verbatim):**")
            if inc.findings_text:
                quote_block(inc.findings_text)
            else:
                st.markdown(badge("missing", NOT_DOCUMENTED), unsafe_allow_html=True)
            field("Court of inquiry", inc.court_of_inquiry)
            field("Historical response and outcome", None)
            for w in inc.warnings:
                st.warning(w)
            source_panel(inc.source_ids, inc.incident_id)

# --------------------------------------------------------------------------- C. findings
with tab_measures:
    st.subheader("Documented preventive findings")
    banner("What these are: after each fatal accident DGMS recorded what would have averted it, in the form "
           "\"Had ... been ..., this accident could have been averted\". Each item below is therefore a precaution "
           "that was NOT in place at the time. The source does not record measures implemented afterwards, their "
           "dates or their effectiveness, and nothing here shows that a precaution is in place today.")
    if not bundle.findings:
        st.info("No preventive finding is documented for this mine's water-related records."
                + (" The historic tables carry no findings." if bundle.incidents else ""))
    else:
        cats = sorted({f.measure_category or "Uncategorised" for f in bundle.findings})
        picked = st.multiselect("Filter by category (assigned by keyword rule, for navigation only)", cats, key="m_cats")
        inc_by_id = {i.incident_id: i for i in bundle.incidents}
        for incident_id in dict.fromkeys(f.incident_id for f in bundle.findings):
            inc = inc_by_id[incident_id]
            items = [f for f in bundle.findings if f.incident_id == incident_id
                     and (not picked or (f.measure_category or "Uncategorised") in picked)]
            if not items:
                continue
            st.markdown(f"#### Accident of {inc.date_text} - {md_escape(inc.cause_label or 'cause not stated')}")
            for f in items:
                with st.container(border=True):
                    st.markdown(
                        badge("doc", "Documented DGMS finding") + badge("missing", "Present status: unknown")
                        + (badge("missing", f.measure_category) if f.measure_category else ""),
                        unsafe_allow_html=True)
                    quote_block("Had " + f.measure_text)
                    a, b = st.columns(2)
                    with a:
                        field("Hazard / accident addressed", f"{inc.cause_label}, {inc.date_text}")
                        field("Implementation date", f.implementation_date)
                        field("Reported outcome or effectiveness", f.reported_outcome)
                    with b:
                        field("Statutory provisions cited", "; ".join(f.regulations_cited))
                        field("Source", "; ".join(sources[s].citation() for s in f.source_ids if s in sources))
                        field("Extraction quality", "separate item in source" if f.extraction_quality == "itemised"
                              else "single paragraph in source (may contain more than one precaution)")
            source_panel(inc.source_ids[:1], f"m_{incident_id}")
    st.caption("Recommendations generated by the system appear only in tab D and are labelled as such.")

# --------------------------------------------------------------------------- D. plan
with tab_plan:
    st.subheader("Emergency response plan (decision support)")
    st.markdown(badge("ai", "System-generated recommendations")
                + "<span class='sk-muted'>Built from the evidence in tabs B and C. Not an official order.</span>",
                unsafe_allow_html=True)
    st.caption(data_flow_notice())
    plans = st.session_state.setdefault("plans", {})
    llm_ready = is_llm_enabled()
    use_llm = st.checkbox("Use the configured language model to write the plan", value=llm_ready, disabled=not llm_ready,
                          help="When off or unavailable, a rule engine assembles the plan from the records.")
    if st.button("Generate plan" if mine.mine_id not in plans else "Regenerate plan", type="primary"):
        try:
            with st.spinner("Retrieving evidence, generating and validating the plan..."):
                plans[mine.mine_id] = get_agent().run(mine.mine_id, use_llm=use_llm)
        except MineNotFound as exc:
            st.error(str(exc))
        except Exception as exc:  # show a recoverable message instead of a stack trace
            st.error(f"The plan could not be generated: {exc}")
            st.caption("The records in the other tabs are unaffected. Check the application log, then try again.")
    result = plans.get(mine.mine_id)
    if result is None:
        st.info("No plan has been generated for this mine in this session.")
    else:
        plan = result.plan
        banner(plan.disclaimer)
        if plan.plan_type == "generic_framework":
            st.warning("GENERIC SAFETY FRAMEWORK. No water-related record for this mine was found in the processed "
                       "documents, so nothing below is specific to this mine's history.")
        if result.llm_error:
            st.warning(f"Language model not used: {result.llm_error}")
        v = plan.validation
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Steps", v.steps_total)
        c2.metric("Backed by this mine's records", v.steps_evidence_backed)
        c3.metric("Statements removed by validation", v.claims_removed)
        c4.metric("Validation", "Passed" if v.passed else "Review needed")
        st.caption(f"Generated by: {plan.generated_by} | {plan.generated_at}")

        st.markdown("#### Historical incident summary")
        st.markdown(md_escape(plan.historical_incident_summary))
        if plan.mine_specific_observations:
            st.markdown("#### Mine-specific observations")
            for o in plan.mine_specific_observations:
                tag = badge("doc", "Documented fact") if o.kind == "documented_fact" else badge("ai", "Inference")
                cites = "; ".join(plan.evidence_citations.get(s, s) for s in o.evidence_source_ids)
                st.markdown(f"{tag}{md_escape(o.text)}" + (f" <span class='sk-muted'>[{cites}]</span>" if cites else ""),
                            unsafe_allow_html=True)

        st.markdown("#### Proposed response steps")
        for stage in plan.stages:
            st.markdown(f"##### Stage {stage.stage_number}: {stage.title}")
            for s in stage.steps:
                with st.container(border=True):
                    basis = (badge("doc", "Based on this mine's records") if s.basis == "documented_evidence"
                             else badge("missing", "General practice - not from this mine's records"))
                    prio = badge("warn" if s.priority == "Immediate" else "missing", f"Priority: {s.priority}")
                    st.markdown(f"**Step {s.step_number}.** {md_escape(s.action)}")
                    st.markdown(badge("ai", "Recommendation") + basis + prio, unsafe_allow_html=True)
                    st.markdown(f"**Why relevant to this mine:** {md_escape(s.relevance_to_this_mine)}")
                    if s.historical_evidence:
                        st.markdown("**Historical evidence:**")
                        quote_block(s.historical_evidence)
                        st.caption("Sources: " + "; ".join(plan.evidence_citations.get(x, x) for x in s.evidence_source_ids))
                    st.markdown(f"**Responsible role:** {md_escape(s.responsible_role)}")
                    if s.verify_before_acting:
                        st.markdown("**Verify before acting:** " + md_escape("; ".join(s.verify_before_acting)))
                    if s.dependencies_and_approvals:
                        st.markdown("**Dependencies and approvals:** " + md_escape("; ".join(s.dependencies_and_approvals)))

        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("#### Missing information")
            for x in plan.missing_information:
                st.markdown(f"- {md_escape(x)}")
            st.markdown("#### Assumptions and uncertainties")
            for x in plan.assumptions_and_uncertainties:
                st.markdown(f"- {md_escape(x)}")
        with b:
            st.markdown("#### Required human verification and approvals")
            for x in plan.required_human_verification_and_approvals:
                st.markdown(f"- {md_escape(x)}")
            st.markdown("#### Evidence citations")
            for sid, cite in plan.evidence_citations.items():
                st.markdown(f"- `{sid}`: {md_escape(cite)}")

        with st.expander(f"Validation report ({len(v.issues)} item(s))"):
            if not v.issues:
                st.write("No unsupported claims, unknown citations or predictive statements were found.")
            for i in v.issues:
                (st.error if i.severity == "error" else st.warning)(
                    f"{i.code} at {i.location}: {i.message} Action: {i.action_taken}.")
        with st.expander("Agent trace (tools called, in order)"):
            st.dataframe(pd.DataFrame(result.trace), hide_index=True, width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button("Download plan (Markdown)", plan_to_markdown(plan),
                           file_name=f"sanket_plan_{mine.mine_id}.md", mime="text/markdown", width="stretch")
        d2.download_button("Download plan (JSON)", plan.model_dump_json(indent=2),
                           file_name=f"sanket_plan_{mine.mine_id}.json", mime="application/json", width="stretch")

# --------------------------------------------------------------------------- search
with tab_search:
    st.subheader("Search this mine's records")
    st.caption("Keyword and semantic search restricted to this mine's records. Results are passages from the "
               "source, not generated text.")
    q = st.text_input("Search terms", key=f"q_{mine.mine_id}", placeholder="e.g. pump, sump, barrier, water level")
    if q:
        hits = retriever.hybrid_search(mine.mine_id, q, limit=8)
        if retriever.vector_error:
            st.caption("Semantic index unavailable; showing keyword matches only.")
        if not hits:
            st.info("No passage in this mine's records matches those terms.")
        for h in hits:
            with st.container(border=True):
                st.markdown(badge("doc", "Source passage")
                            + f"<span class='sk-muted'>{h.source.citation() if h.source else ''} | "
                              f"matched by {', '.join(h.matched_by)}</span>", unsafe_allow_html=True)
                st.text(h.text)
