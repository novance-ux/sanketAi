"""Chat UI: the dedicated Bot page (full conversational workspace) and a corner shortcut to it.

Layout of the Bot page:
* left column  - new conversation, previous conversations (open, rename, delete, clear), memory controls;
* main column  - answer mode (Research report / Quick answer), a tall conversation area where each
  question is followed by its answer and the newest message stays in view, and the input box pinned at
  the bottom of the page.

Research reports show, under the written assessment: charts built from PDF data, the PDF fact-check
table, web and PDF sources (links, dates, page numbers), and the research steps the agent took.
Answers state what was actually checked; document text, web text and model output render as Markdown
without HTML.
"""
from __future__ import annotations

import html
import math
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .chat import format_answer
from .chat_service import ChatService
from .chat_store import (
    CHECKED,
    CLARIFY,
    ERROR,
    INSUFFICIENT,
    NOT_CHECKED,
    PARTIAL,
    PASSAGES_ONLY,
    PENDING,
    RESEARCH,
    STATUS_LABELS,
    UNVERIFIED,
    VERIFIED,
    NotFound,
    new_user_token,
    owner_id_for,
)
from .config import get_settings

STATUS_BADGE = {CHECKED: "doc", NOT_CHECKED: "missing", VERIFIED: "doc", PARTIAL: "ai", INSUFFICIENT: "missing",
                UNVERIFIED: "warn", PASSAGES_ONLY: "missing", PENDING: "warn", ERROR: "warn", RESEARCH: "doc",
                CLARIFY: "ai"}
VERDICT_LABELS = {"supported": "Supported by the PDFs", "contradicted": "Contradicted by the PDFs",
                  "qualified": "Partly supported / qualified", "not_addressed": "Not addressed in the PDFs"}
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical palette (colour-blind safe)
MODES = {"research": "Research report", "quick": "Quick answer", "pdf": "PDF answer"}
BOT_PAGE = "pages/0_Bot.py"


@st.cache_resource(show_spinner=False)
def get_chat_service() -> ChatService:
    return ChatService(get_settings())


def _badge(kind: str, text: str) -> str:
    return f'<span class="sk-badge sk-{kind}">{html.escape(text)}</span>'


def _plain(text) -> str:
    """Escape Markdown so partial or untrusted text renders literally."""
    return re.sub(r"([\\`*_{}\[\]<>#$|~])", r"\\\1", str(text or ""))


def _safe_url(url: str) -> str | None:
    return url if url.startswith(("https://", "http://")) and not re.search(r"[\s()<>\"']", url) else None


def current_owner() -> str:
    """Anonymous but unguessable identity, kept in the session and in the URL (?u=...) across reloads."""
    token = st.session_state.get("chat_user_token") or st.query_params.get("u")
    if not token or len(token) < 16:
        token = new_user_token()
    st.session_state["chat_user_token"] = token
    if st.query_params.get("u") != token:
        st.query_params["u"] = token
    return owner_id_for(token)


# --------------------------------------------------------------------------- corner shortcut
def render_floating_chat() -> None:
    """Small button in the bottom-right corner of every other page that opens the Bot page."""
    with st.container(key="sanket_fab"):
        st.page_link(BOT_PAGE, label="Inundation chat", icon=":material/chat:", help="Open Inundation chat")


# --------------------------------------------------------------------------- Bot page
SUGGESTIONS = [
    ("I am placed as CMD at ECL. What are the inundation conditions there?", ":material/travel_explore:"),
    ("What happened at Sayal D colliery in 2017 and what could have prevented it?", ":material/history_edu:"),
    ("Which coal mines had irruption of water accidents since 1901?", ":material/water_drop:"),
    ("How many fatal accidents occurred in coal mines in 2019?", ":material/bar_chart:"),
]


def render_bot_page() -> None:
    """ChatGPT-style workspace: conversations in the sidebar, a centred mode switch, and either a centred
    "Where should we begin?" start screen or the full conversation with the input pinned at the bottom."""
    try:
        service = get_chat_service()
    except Exception as exc:
        st.error(f"The chat could not start: {exc}")
        return
    owner = current_owner()
    conv_id = st.session_state.get("chat_conv_id")
    if conv_id:
        try:
            service.store.get_conversation(owner, conv_id)
        except NotFound:
            conv_id = None
            st.session_state.pop("chat_conv_id", None)

    with st.sidebar:
        st.page_link("app.py", label="Back to main menu", icon=":material/arrow_back:")
        if st.button("New chat", key="bot_new", icon=":material/edit_square:", width="stretch"):
            st.session_state.pop("chat_conv_id", None)
            st.rerun()
        st.caption("Recents")
        _history_list(service, owner, conv_id)
        with st.expander("Memory", icon=":material/psychology:"):
            _memory_controls(service, owner)

    messages = service.store.messages(owner, conv_id) if conv_id else []
    pending = st.session_state.pop("bot_pending", None)

    if not messages and not pending:
        st.markdown('<div class="sk-hero">Where should we begin?</div>', unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 3, 1])
        with mid:
            prompt, mode = _input_row()
            for n, (text, icon) in enumerate(SUGGESTIONS):
                if st.button(text, key=f"suggest_{n}", icon=icon, type="tertiary"):
                    st.session_state["bot_pending"] = text
                    st.rerun()
        if not prompt:
            return
        box = st.container(height=680, key="bot_messages", autoscroll=True)
    else:
        box = st.container(height=680, key="bot_messages", autoscroll=True)
        with box:
            for i, m in enumerate(messages):
                _render_message(m, i)
            if messages and messages[-1]["role"] == "assistant" and messages[-1]["status"] in (PENDING, ERROR):
                last_q = next((x["content"] for x in reversed(messages) if x["role"] == "user"), None)
                if last_q and st.button("Retry this question", key="bot_retry", icon=":material/refresh:"):
                    st.session_state["bot_pending"] = last_q
                    st.rerun()
        _, mid, _ = st.columns([1, 10, 1])
        with mid:
            prompt, mode = _input_row()
        prompt = prompt or pending
    if prompt:
        with box:
            _stream(service, owner, prompt, conv_id, mode)
        st.rerun()


def _input_row() -> tuple[str | None, str]:
    """Message box with the answer-mode dropdown next to the send arrow."""
    box_col, mode_col = st.columns([6, 1.2], vertical_alignment="bottom", gap="small")
    with mode_col:
        mode = st.selectbox("Answer mode", list(MODES), format_func=MODES.get, key="bot_mode",
                            label_visibility="collapsed",
                            help="Research report: web research fact-checked against the PDFs. "
                                 "Quick answer: from the internet. PDF answer: from the two SANKET PDFs first.")
    with box_col:
        prompt = st.chat_input(f"Ask Inundation - {MODES[mode].lower()}", key="bot_input",
                               max_chars=get_settings().chat_max_message_chars)
    return prompt, mode


def _history_list(service: ChatService, owner: str, current: str | None) -> None:
    convs = service.store.list_conversations(owner)
    if not convs:
        st.caption("No conversations yet.")
        return
    for c in convs:
        a, b, d = st.columns([6, 1, 1], vertical_alignment="center", gap="small")
        label = c["title"] if len(c["title"]) <= 38 else c["title"][:36] + "..."
        if a.button(label, key=f"open_{c['id']}", width="stretch",
                    type="primary" if c["id"] == current else "tertiary", help=c["title"]):
            st.session_state["chat_conv_id"] = c["id"]
            st.rerun()
        with b.popover("", icon=":material/edit:", help="Rename"):
            new = st.text_input("New title", value=c["title"], key=f"title_{c['id']}", max_chars=80)
            if st.button("Save", key=f"rename_{c['id']}"):
                try:
                    service.store.rename_conversation(owner, c["id"], new)
                except ValueError as exc:
                    st.error(str(exc))
                st.rerun()
        if d.button("", key=f"del_{c['id']}", icon=":material/delete:", help="Delete this conversation",
                    type="tertiary"):
            service.store.delete_conversation(owner, c["id"])
            if st.session_state.get("chat_conv_id") == c["id"]:
                st.session_state.pop("chat_conv_id", None)
            st.rerun()
    confirm = st.checkbox("Delete every conversation", key="confirm_clear")
    if st.button("Clear all history", key="clear_all", disabled=not confirm, width="stretch"):
        service.store.clear_history(owner)
        st.session_state.pop("chat_conv_id", None)
        st.rerun()


def _memory_controls(service: ChatService, owner: str) -> None:
    st.caption("Short topic notes from your own earlier questions, used only when you refer back to an "
               "earlier conversation. Context only, never evidence.")
    enabled = service.store.memory_enabled(owner)
    new = st.toggle("Use memory of earlier conversations", value=enabled, key="mem_toggle")
    if new != enabled:
        service.store.set_memory_enabled(owner, new)
        st.rerun()
    memories = service.store.memories(owner)
    for m in memories:
        st.markdown(f"**{_plain(m['title'])}**")
        st.caption(m["summary"])
    if memories and st.button("Delete all memory notes", key="mem_delete", icon=":material/delete:"):
        service.store.delete_memories(owner)
        st.rerun()


# --------------------------------------------------------------------------- streaming
def _stream(service: ChatService, owner: str, prompt: str, conv_id: str | None, mode: str) -> None:
    with st.chat_message("user"):
        st.markdown(_plain(prompt))
    with st.chat_message("assistant"):
        progress = st.status("Researching..." if mode == "research" else "Answering...", expanded=mode == "research")
        answer_slot = st.empty()
        st.button("Stop", key="bot_stop", icon=":material/stop_circle:", type="tertiary",
                  help="Stop this answer. The question is kept and can be retried.")
        text = ""
        for ev in service.ask(owner, prompt, conv_id, None, mode):
            if ev.type == "status":
                if ev.data.get("conversation_id"):
                    st.session_state["chat_conv_id"] = ev.data["conversation_id"]
                if ev.data.get("reset_draft"):
                    text = ""
                progress.update(label=ev.data.get("message", ev.data["stage"]) + "...")
                if ev.data["stage"] not in ("started",):
                    progress.write(ev.data.get("message", ev.data["stage"]))
            elif ev.type == "token":
                text += ev.data["text"]
                answer_slot.markdown(format_answer(text) + " ▌")
            elif ev.type == "error":
                progress.update(label="Could not answer", state="error")
                answer_slot.error(ev.data["message"])
                if ev.data.get("recoverable") and st.button("Retry", key="bot_retry_err", icon=":material/refresh:"):
                    st.session_state["bot_pending"] = prompt
                    st.rerun()
                st.stop()
            elif ev.type == "done":
                progress.update(label="Done", state="complete", expanded=False)
                break


# --------------------------------------------------------------------------- message rendering
def _render_message(m: dict, index: int) -> None:
    if m["role"] == "user":
        with st.chat_message("user"):
            st.markdown(_plain(m["content"]))
        return
    meta = m.get("meta") or {}
    status = m["status"]
    with st.chat_message("assistant"):
        if status == PENDING:
            st.caption("Stopped or interrupted before the answer was finished.")
            return
        st.markdown(format_answer(m["content"]))
        if meta.get("mode") == "research":
            _render_research_extras(meta, index)
        st.markdown(_badge(STATUS_BADGE.get(status, "missing"), STATUS_LABELS.get(status, status)),
                    unsafe_allow_html=True)
        cites = meta.get("citations") or []
        if cites:
            _render_sources(cites, meta)
        v = meta.get("verification") or {}
        if v and (v.get("removed") or v.get("checker_used")):
            with st.expander("How this answer was checked"):
                st.caption("Every statement was checked by an independent model call against the sources."
                           if v.get("checker_used") else
                           "Citations were checked to exist, and every number was checked to appear in the cited "
                           "source. The meaning of each statement was not checked by a second model.")
                for r in v.get("removed", []):
                    st.markdown(f"- Removed: {_plain(r['claim'])} *({_plain(r['reason'])})*")
        extra = []
        if meta.get("cached"):
            extra.append("reused from cache")
        if meta.get("web_error"):
            extra.append(meta["web_error"])
        if extra:
            st.caption(" | ".join(extra))


def _render_research_extras(meta: dict, index: int) -> None:
    for n, chart in enumerate(meta.get("charts") or []):
        _render_chart(chart, f"{index}_{n}")
    checks = meta.get("fact_check") or []
    if checks:
        st.markdown("**PDF fact-check of the key web findings**")
        st.dataframe(pd.DataFrame([{
            "Claim from internet research": c["claim"],
            "Web source": ", ".join(c["web"]),
            "Result against the SANKET PDFs": VERDICT_LABELS.get(c["verdict"], c["verdict"]),
            "PDF reference": "; ".join(c.get("pdf_pages") or []) or "-",
            "Note": c.get("note") or "",
        } for c in checks]), hide_index=True, width="stretch")
    flags = meta.get("flags") or []
    if flags:
        with st.expander(f"Figures to double-check ({len(flags)})"):
            st.caption("These sentences contain numbers that were not found in the source they cite.")
            for f in flags:
                st.markdown(f"- {_plain(f)}")
    u = meta.get("understanding") or {}
    if u:
        with st.expander("Research steps"):
            ctx = u.get("user_context") or {}
            st.markdown(
                f"- **Question understood as:** {_plain(u.get('restated_question') or '')}\n"
                f"- **Role / context:** {_plain(ctx.get('role_expanded') or ctx.get('role') or 'not stated')}"
                + (f" (age {_plain(ctx['age'])}, context only)" if ctx.get("age") else "") + "\n"
                f"- **Organisation:** {_plain(u.get('organization_full_name') or u.get('organization_abbreviation') or 'none identified')}\n"
                f"- **Places / mines:** {_plain(', '.join(u.get('locations', []) + u.get('mines', [])) or 'none named')}\n"
                f"- **Scope assessed:** {_plain(u.get('scope') or '-')}\n"
                f"- **PDF owner match:** {_plain(', '.join(meta.get('owners') or []) or 'none')}")
            if u.get("assumptions"):
                st.markdown("**Assumptions (not stated by you):** " + _plain("; ".join(u["assumptions"])))
            st.markdown("**Internet searches:** " + _plain(" | ".join(u.get("web_queries", []))))
            for note in meta.get("notes") or []:
                st.caption(note)


def _render_chart(chart: dict, key: str) -> None:
    if chart["kind"] == "table":
        st.markdown(f"**{_plain(chart['title'])}**")
        st.dataframe(pd.DataFrame(chart["rows"], columns=chart["columns"]), hide_index=True, width="stretch")
        st.caption(f"Source: {chart['source']}. {chart.get('note', '')}")
        return
    fig = go.Figure()
    for n, (name, values) in enumerate(chart["series"].items()):
        fig.add_trace(go.Bar(x=[str(x) for x in chart["x"]], y=values, name=name,
                             marker_color=SERIES_COLORS[n % len(SERIES_COLORS)],
                             hovertemplate=f"{name}<br>%{{x}}: %{{y}}<extra></extra>"))
    top = max((max(v) for v in chart["series"].values() if v), default=1)
    fig.update_layout(
        title=dict(text=chart["title"], font=dict(size=14)), barmode="group", height=320,
        margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        font=dict(color="#52514e", size=12), legend=dict(orientation="h", y=-0.2), bargap=0.25,
        xaxis=dict(title=chart["x_label"], type="category", linecolor="#c3c2b7"),
        yaxis=dict(title=chart["y_label"], gridcolor="#ecebe6", dtick=1 if top <= 10 else None,
                   range=[0, math.ceil(top * 1.15) + 0.5]))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"chart_{key}")
    st.caption(f"Source: {chart['source']}. {chart.get('note', '')}")


def _render_sources(cites: list[dict], meta: dict) -> None:
    pdf = [c for c in cites if c.get("kind") != "web"]
    web = [c for c in cites if c.get("kind") == "web"]
    dates = {w["sid"]: w.get("date") for w in meta.get("web_sources") or []}
    if meta.get("mode") == "research":
        label = f"Sources: {len(web)} from the internet, {len(pdf)} from the SANKET PDFs"
    elif meta.get("mode") == "quick_web":
        label = f"Sources: {len(web)} from the internet"
    else:
        label = f"Sources: {len(pdf)} from the PDFs" + (f", {len(web)} from the web" if web else "")
    with st.expander(label):
        for c in web:
            url = _safe_url(c.get("url") or "")
            title = _plain(c.get("source_filename") or "web page")
            when = dates.get(c["sid"]) or "date not stated"
            st.markdown(f"**[{c['sid']}]** " + (f"[{title}]({url})" if url else title) + f" - {_plain(when)}")
            st.text(c["excerpt"][:700])
        for c in pdf:
            st.markdown(f"**[{c['sid']}]** {_plain(c['citation'])}" + (f" | {_plain(c['mine'])}" if c.get("mine") else ""))
            st.text(c["excerpt"][:1500] + ("..." if len(c["excerpt"]) > 1500 else ""))
