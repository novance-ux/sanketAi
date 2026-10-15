"""Chat UI: the floating chatbot shown in the bottom-right corner of every page.

The panel runs inside a Streamlit fragment, so sending a message, opening history or closing the
panel reruns only the panel - the page underneath keeps its state. Conversations live in the chat
database (src/chat_store.py), so closing the panel or reloading the page loses nothing.

Display rules:
* Messages read like any chat: each question (right) followed by its answer (left), newest at the
  bottom, scrolled into view.
* The answer streams in as it is written, then is replaced by the checked version, labelled with what
  was actually checked ("Citations checked", or "Sources verified" in thorough mode).
* Document text, web snippets and model output are rendered as Markdown without HTML.
"""
from __future__ import annotations

import html
import re

import streamlit as st

from .chat import format_answer
from .chat_service import ChatService
from .chat_store import (
    CHECKED,
    ERROR,
    NOT_CHECKED,
    INSUFFICIENT,
    PARTIAL,
    PASSAGES_ONLY,
    PENDING,
    STATUS_LABELS,
    UNVERIFIED,
    VERIFIED,
    NotFound,
    new_user_token,
    owner_id_for,
)
from .config import get_settings

STATUS_BADGE = {CHECKED: "doc", NOT_CHECKED: "missing", VERIFIED: "doc", PARTIAL: "ai", INSUFFICIENT: "missing", UNVERIFIED: "warn",
                PASSAGES_ONLY: "missing", PENDING: "warn", ERROR: "warn"}


@st.cache_resource(show_spinner=False)
def get_chat_service() -> ChatService:
    return ChatService(get_settings())


def _badge(kind: str, text: str) -> str:
    return f'<span class="sk-badge sk-{kind}">{html.escape(text)}</span>'


def _plain(text: str) -> str:
    """Escape Markdown so partial or untrusted text renders literally."""
    return re.sub(r"([\\`*_{}\[\]<>#$|~])", r"\\\1", text or "")


def _rerun() -> None:
    """Rerun only the chat panel when possible (inside a fragment rerun), else the whole page."""
    from streamlit.errors import StreamlitAPIException

    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        st.rerun()


def current_owner() -> str:
    """Anonymous but unguessable identity, kept in the session and in the URL (?u=...) across reloads."""
    token = st.session_state.get("chat_user_token") or st.query_params.get("u")
    if not token or len(token) < 16:
        token = new_user_token()
    st.session_state["chat_user_token"] = token
    if st.query_params.get("u") != token:
        st.query_params["u"] = token
    return owner_id_for(token)


# --------------------------------------------------------------------------- entry points
def render_floating_chat() -> None:
    _chat_fragment("floating")


@st.fragment
def _chat_fragment(mode: str) -> None:
    try:
        service = get_chat_service()
    except Exception:  # the chat must never break the page it floats over
        return
    owner = current_owner()
    is_open = st.session_state.get("chat_open", False)
    with st.container(key="sanket_fab"):
        if st.button("Close" if is_open else "Ask SANKET", key="sanket_fab_btn",
                     icon=":material/close:" if is_open else ":material/chat:",
                     help="Ask questions about the two SANKET volumes"):
            st.session_state["chat_open"] = not is_open
            _rerun()
    if not is_open:
        return
    with st.container(key="sanket_chat_panel"):
        _panel(service, owner, mode)


# --------------------------------------------------------------------------- panel
def _panel(service: ChatService, owner: str, mode: str) -> None:
    view = st.session_state.get("chat_view", "chat")
    conv_id = st.session_state.get("chat_conv_id")
    if conv_id:
        try:
            conv = service.store.get_conversation(owner, conv_id)
        except NotFound:
            conv, conv_id = None, None
            st.session_state.pop("chat_conv_id", None)
    else:
        conv = None

    head, b1, b2, b3 = st.columns([5, 1, 1, 1], vertical_alignment="center", gap="small")
    head.markdown(f"**{html.escape(conv['title']) if conv else 'Ask SANKET'}**")
    if b1.button("", key=f"chat_new_{mode}", icon=":material/add_comment:", help="New conversation"):
        st.session_state.pop("chat_conv_id", None)
        st.session_state["chat_view"] = "chat"
        _rerun()
    if b2.button("", key=f"chat_hist_{mode}", icon=":material/history:", help="Previous conversations"):
        st.session_state["chat_view"] = "chat" if view == "history" else "history"
        _rerun()
    if b3.button("", key=f"chat_mem_{mode}", icon=":material/psychology:", help="Memory settings"):
        st.session_state["chat_view"] = "chat" if view == "memory" else "memory"
        _rerun()

    if view == "history":
        _history_view(service, owner, mode)
    elif view == "memory":
        _memory_view(service, owner, mode)
    else:
        _chat_view(service, owner, mode, conv_id)


def _history_view(service: ChatService, owner: str, mode: str) -> None:
    st.caption("Previous conversations (stored on this machine for your session link).")
    convs = service.store.list_conversations(owner)
    if not convs:
        st.info("No conversations yet.")
    for c in convs:
        a, b, d = st.columns([6, 1, 1], vertical_alignment="center")
        if a.button(f"{c['title']}  ({c['message_count'] // 2} Q)", key=f"open_{mode}_{c['id']}",
                    width="stretch", help=f"Last activity {c['updated_at']}"):
            st.session_state["chat_conv_id"] = c["id"]
            st.session_state["chat_view"] = "chat"
            _rerun()
        with b.popover("", icon=":material/edit:", help="Rename"):
            new = st.text_input("New title", value=c["title"], key=f"title_{mode}_{c['id']}", max_chars=80)
            if st.button("Save", key=f"rename_{mode}_{c['id']}"):
                try:
                    service.store.rename_conversation(owner, c["id"], new)
                except ValueError as exc:
                    st.error(str(exc))
                _rerun()
        if d.button("", key=f"del_{mode}_{c['id']}", icon=":material/delete:", help="Delete this conversation"):
            service.store.delete_conversation(owner, c["id"])
            if st.session_state.get("chat_conv_id") == c["id"]:
                st.session_state.pop("chat_conv_id", None)
            _rerun()
    if convs:
        st.divider()
        confirm = st.checkbox("I want to delete every conversation", key=f"confirm_clear_{mode}")
        if st.button("Clear all history", key=f"clear_all_{mode}", disabled=not confirm, type="secondary"):
            service.store.clear_history(owner)
            st.session_state.pop("chat_conv_id", None)
            st.session_state["chat_view"] = "chat"
            _rerun()


def _memory_view(service: ChatService, owner: str, mode: str) -> None:
    st.caption("Long-term memory: short topic notes built from your own earlier questions. They help the "
               "assistant understand follow-ups across conversations. They are context only, never evidence: "
               "every answer is still checked against the PDFs.")
    enabled = service.store.memory_enabled(owner)
    new = st.toggle("Use memory of earlier conversations", value=enabled, key=f"mem_toggle_{mode}")
    if new != enabled:
        service.store.set_memory_enabled(owner, new)
        _rerun()
    memories = service.store.memories(owner)
    if not memories:
        st.info("Nothing is stored in memory.")
    for m in memories:
        st.markdown(f"**{html.escape(m['title'])}**")
        st.caption(m["summary"])
    if memories and st.button("Delete all memory notes", key=f"mem_delete_{mode}", icon=":material/delete:"):
        service.store.delete_memories(owner)
        _rerun()


def _safe_url(url: str) -> str | None:
    return url if url.startswith(("https://", "http://")) and not re.search(r"[\s()<>\"']", url) else None


def _render_sources(cites: list[dict]) -> None:
    pdf = [c for c in cites if c.get("kind") != "web"]
    web = [c for c in cites if c.get("kind") == "web"]
    label = f"Sources: {len(pdf)} from the PDFs" + (f", {len(web)} from the web" if web else "")
    with st.expander(label):
        for c in pdf:
            st.markdown(f"**[{c['sid']}]** {_plain(c['citation'])}" + (f" | {_plain(c['mine'])}" if c.get("mine") else ""))
            st.text(c["excerpt"][:1500] + ("..." if len(c["excerpt"]) > 1500 else ""))
        if web:
            st.caption("Internet sources (secondary; not part of the SANKET documents)")
        for c in web:
            url = _safe_url(c.get("url") or "")
            title = _plain(c.get("source_filename") or "web page")
            st.markdown(f"**[{c['sid']}]** " + (f"[{title}]({url})" if url else title))
            st.text(c["excerpt"][:600])


def _render_message(m: dict, mode: str, index: int) -> None:
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
        cites = meta.get("citations") or []
        st.markdown(_badge(STATUS_BADGE.get(status, "missing"), STATUS_LABELS.get(status, status)),
                    unsafe_allow_html=True)
        if cites:
            _render_sources(cites)
        v = meta.get("verification") or {}
        if v and (v.get("removed") or v.get("checker_used")):
            with st.expander("How this answer was checked"):
                if v.get("checker_used"):
                    st.caption("Every statement was checked by an independent model call against the sources.")
                else:
                    st.caption("Citations were checked to exist, and every number was checked to appear in the "
                               "cited source. The meaning of each statement was not checked by a second model.")
                for r in v.get("removed", []):
                    st.markdown(f"- Removed: {_plain(r['claim'])} *({_plain(r['reason'])})*")
        extra = []
        if meta.get("answered_by"):
            extra.append(f"Answered by {meta['answered_by']}")
        if meta.get("cached"):
            extra.append("reused from cache")
        if meta.get("web_error"):
            extra.append(meta["web_error"])
        if extra:
            st.caption(" | ".join(extra))


def _chat_view(service: ChatService, owner: str, mode: str, conv_id: str | None) -> None:
    category = None  # both volumes
    messages = service.store.messages(owner, conv_id) if conv_id else []
    # newest message at the bottom, scrolled into view, like any chat window
    box = st.container(height=400, key=f"chat_messages_{mode}", autoscroll=True)
    with box:
        for i, m in enumerate(messages):
            _render_message(m, mode, i)
        if messages and messages[-1]["role"] == "assistant" and messages[-1]["status"] in (PENDING, ERROR):
            last_q = next((x["content"] for x in reversed(messages) if x["role"] == "user"), None)
            if last_q and st.button("Retry this question", key=f"retry_{mode}", icon=":material/refresh:"):
                st.session_state[f"chat_pending_{mode}"] = last_q
                _rerun()

    prompt = st.chat_input("Ask about the SANKET documents", key=f"chat_input_{mode}",
                           max_chars=get_settings().chat_max_message_chars)
    prompt = prompt or st.session_state.pop(f"chat_pending_{mode}", None)
    if not prompt:
        return

    with box:
        with st.chat_message("user"):
            st.markdown(_plain(prompt))
        with st.chat_message("assistant"):
            answer_slot = st.empty()
            status_slot = st.empty()
            answer_slot.caption("Thinking...")
            st.button("Stop", key=f"chat_stop_{mode}", icon=":material/stop_circle:", type="tertiary",
                      help="Stop this answer. The question is kept and can be retried.")
            text = ""
            for ev in service.ask(owner, prompt, conv_id, category):
                if ev.type == "status":
                    if ev.data.get("conversation_id"):
                        st.session_state["chat_conv_id"] = ev.data["conversation_id"]
                    if ev.data.get("reset_draft"):
                        text = ""
                        answer_slot.caption("Rewriting the answer...")
                    status_slot.caption(ev.data.get("message", ev.data["stage"]) + "...")
                elif ev.type == "token":
                    text += ev.data["text"]
                    answer_slot.markdown(format_answer(text) + " ▌")
                elif ev.type == "error":
                    status_slot.empty()
                    answer_slot.error(ev.data["message"])
                    if ev.data.get("recoverable") and st.button("Retry", key=f"retry_err_{mode}",
                                                                icon=":material/refresh:"):
                        st.session_state[f"chat_pending_{mode}"] = prompt
                        _rerun()
                    return
                elif ev.type == "done":
                    break
    _rerun()
