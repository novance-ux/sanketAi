"""Verified chat pipeline: storage, cache, claim verification, event stream, HTTP API and floating UI.

Model calls are replaced by deterministic fakes (the drafting stream and the claim checker), so these
tests exercise the real retrieval, verification logic, persistence and UI without network access.
The fakes stand in only for the language model; citations and passages come from the real index.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import chat as chat_module
from src import chat_service as service_module
from src import verification as verification_module
from src.chat import Passage
from src.chat_cache import SqliteCache, make_key
from src.chat_service import ChatService, RateLimiter
from src.chat_store import (
    INSUFFICIENT,
    PARTIAL,
    PASSAGES_ONLY,
    PENDING,
    UNVERIFIED,
    VERIFIED,
    ChatStore,
    NotFound,
    new_user_token,
    owner_id_for,
)
from src.config import get_settings
from src.llm import LLMUnavailable
from src.verification import split_claims, verify_answer

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- fakes
def fake_checker(system, messages, settings=None, json_mode=False, used=None):
    """Stands in for the independent checker model: rejects claims containing 'invented' or '9999'."""
    body = messages[-1][1]
    if "CLAIMS:" not in body:  # query-rewrite call
        return "Sayal D colliery 2017 regulation cited in DGMS finding"
    claims = re.findall(r"^(\d+)\. (.*)$", body.split("CLAIMS:")[1], re.M)
    return json.dumps({"results": [
        {"id": int(i), "verdict": "unsupported" if ("invented" in t or "9999" in t) else "supported",
         "evidence": ["S1"], "reason": "test"} for i, t in claims]})


class FakeDraft:
    """Stands in for the drafting model; returns scripted drafts in order."""

    def __init__(self, *drafts: str):
        self.drafts = list(drafts)
        self.calls: list[list] = []

    def __call__(self, system, messages, settings=None, used=None):
        self.calls.append(messages)
        text = self.drafts.pop(0) if len(self.drafts) > 1 else self.drafts[0]
        if used is not None:
            used.update(provider="fake", model="fake-model", skipped=[])
        for i in range(0, len(text), 25):
            yield text[i:i + 25]


@pytest.fixture()
def chat_env(real_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("SANKET_CHAT_DIR", str(tmp_path / "chat"))
    monkeypatch.setenv("SANKET_REDIS_URL", "")
    monkeypatch.setenv("SANKET_CHAT_VERIFY_MODE", "thorough")  # most tests here exercise the full check
    monkeypatch.setenv("SANKET_CHAT_WEB", "false")
    monkeypatch.setattr(service_module, "is_llm_enabled", lambda s=None: True)
    monkeypatch.setattr(verification_module, "complete_text", fake_checker)
    monkeypatch.setattr(service_module, "complete_text", fake_checker)
    return get_settings()


def run(service: ChatService, owner: str, question: str, conversation_id=None, category=None):
    events = list(service.ask(owner, question, conversation_id, category))
    done = next((e for e in events if e.type == "done"), None)
    return events, done


GOOD = ("In 2019 the coal volume records 51 fatal accidents in coal mines with 56 fatalities and 6 seriously "
        "injured persons [S1].")


# --------------------------------------------------------------------------- store
def test_conversations_are_isolated_per_user(tmp_path):
    store = ChatStore(tmp_path / "c.sqlite3")
    alice, bob = owner_id_for(new_user_token()), owner_id_for(new_user_token())
    conv = store.create_conversation(alice, "Alice's chat")
    store.add_message(alice, conv["id"], "user", "hello there", "received")
    for action in (lambda: store.get_conversation(bob, conv["id"]), lambda: store.messages(bob, conv["id"]),
                   lambda: store.rename_conversation(bob, conv["id"], "x"),
                   lambda: store.delete_conversation(bob, conv["id"]),
                   lambda: store.add_message(bob, conv["id"], "user", "x", "received")):
        with pytest.raises(NotFound):
            action()
    assert store.list_conversations(bob) == [] and store.clear_history(bob) == 0
    assert len(store.messages(alice, conv["id"])) == 1
    with pytest.raises(ValueError):
        owner_id_for("short")


def test_conversation_crud_and_memory_controls(tmp_path):
    store = ChatStore(tmp_path / "c.sqlite3")
    me = owner_id_for(new_user_token())
    c1 = store.create_conversation(me)
    store.add_message(me, c1["id"], "user", "What happened at Kuardih colliery in 2017?", "received")
    store.add_message(me, c1["id"], "assistant", "A model claim that must not enter memory.", VERIFIED)
    store.autotitle(me, c1["id"], "What happened at Kuardih colliery in 2017?")
    store.refresh_summary(me, c1["id"])
    assert store.get_conversation(me, c1["id"])["title"].startswith("What happened at Kuardih")
    note = store.memories(me)[0]["summary"]
    assert "Kuardih" in note and "model claim" not in note
    assert store.relevant_memories(me, "tell me more about the Kuardih inrush", None)
    assert store.relevant_memories(me, "Kuardih", exclude_conversation=c1["id"]) == []
    store.set_memory_enabled(me, False)
    assert store.relevant_memories(me, "Kuardih colliery", None) == []
    store.delete_memories(me)
    assert store.memories(me) == []
    store.rename_conversation(me, c1["id"], "Renamed")
    assert store.list_conversations(me)[0]["title"] == "Renamed"
    store.delete_conversation(me, c1["id"])
    assert store.list_conversations(me) == []


# --------------------------------------------------------------------------- cache
def test_cache_keys_and_invalidation_follow_the_document_version(tmp_path):
    k1 = make_key("answer", "v1", "q", None)
    assert k1 != make_key("answer", "v2", "q", None) and k1 != make_key("answer", "v1", "q2", None)
    cache = SqliteCache(tmp_path / "cache.sqlite3", "v1")
    cache.set(k1, {"a": 1}, 3600)
    assert cache.get(k1) == {"a": 1}
    reopened = SqliteCache(tmp_path / "cache.sqlite3", "v2")  # a PDF changed -> new version
    assert reopened.size() == 0 and reopened.get(k1) is None


# --------------------------------------------------------------------------- verification
def test_claims_are_split_and_typed():
    claims = split_claims(
        "- The wall failed under Regulation 129(1) read with D.G.'s Circular No. 7 of 1976. Two persons died [S1].\n"
        "- DGMS enforces the Mines Act [GK].\n- This suggests weak barriers [INF][S1].\n"
        "The two SANKET volumes provided do not contain the current water level.")
    kinds = [c.kind for c in claims]
    assert kinds == ["doc", "doc", "general", "inference", "refusal"]
    assert "Circular No. 7 of 1976" in claims[0].text and claims[0].cited == ["S1"]


def test_unsupported_and_invented_figures_are_removed(chat_env):
    passages = [Passage("S1", "page", "vol.pdf", [159], "The number of fatal accidents in 2019 stands at 51 "
                                                         "with 56 fatalities and 6 seriously injured.")]
    draft = ("There were 51 fatal accidents and 56 fatalities in 2019 [S1]. "
             "The death toll later rose to 9999 according to the report [S1]. "
             "An invented statement about pumps failing everywhere [S1].")
    result = verify_answer(draft, passages, chat_env)
    assert result.status == PARTIAL and result.checker_used
    assert "51 fatal accidents" in result.answer and "9999" not in result.answer and "invented" not in result.answer
    reasons = " ".join(c.removal_reason for c in result.removed)
    assert "number(s) not found" in reasons and "checker verdict" in reasons


def test_bad_citations_and_uncited_claims(chat_env):
    passages = [Passage("S1", "page", "vol.pdf", [1], "Kuardih inrush of water in 2017 drowned one person.")]
    result = verify_answer("An inrush of water at Kuardih drowned one person [S7]. "
                           "The inrush happened in 2017 at Kuardih colliery.", passages, chat_env)
    kept = [c for c in result.claims if c.keep]
    assert all(c.cited == ["S1"] for c in kept) and "[S1]" in result.answer and "[S7]" not in result.answer


def test_refusal_is_reported_as_insufficient_evidence_without_citations(chat_env):
    passages = [Passage("S1", "page", "vol.pdf", [304], "Chasnalla 1975 irruption of water 375 killed.")]
    result = verify_answer("The two SANKET volumes provided do not contain the current water level at "
                           "Chasnalla [S1].", passages, chat_env)
    assert result.status == INSUFFICIENT and "[S1]" not in result.answer and not result.needs_regeneration


def test_general_knowledge_is_capped(chat_env):
    passages = [Passage("S1", "page", "vol.pdf", [1], "Two persons drowned in the sump.")]
    draft = ("Two persons drowned in the sump [S1]. Drowning is a leading cause of death in flooded mines around "
             "the world [GK]. Sumps collect water pumped from mine workings and need fencing [GK].")
    result = verify_answer(draft, passages, chat_env)
    assert result.general_share <= 0.2 + 1e-9
    assert any("above the 20% share" in c.removal_reason for c in result.removed)


def test_checker_unavailable_is_labelled_unverified(chat_env, monkeypatch):
    def down(*a, **k):
        raise LLMUnavailable("checker offline")

    monkeypatch.setattr(verification_module, "complete_text", down)
    passages = [Passage("S1", "page", "vol.pdf", [159], "fatal accidents in 2019 stands at 51 with 56 fatalities")]
    result = verify_answer("Fatal accidents in 2019 stand at 51 with 56 fatalities [S1].", passages, chat_env)
    assert result.status == UNVERIFIED and not result.checker_used and "checker offline" in result.checker_error


def test_nothing_supported_requests_regeneration(chat_env):
    passages = [Passage("S1", "page", "vol.pdf", [1], "unrelated text")]
    result = verify_answer("An invented accident happened [S1].", passages, chat_env)
    assert result.status == INSUFFICIENT and result.needs_regeneration
    assert verify_answer("An invented accident happened [S1].", passages, chat_env,
                         allow_regeneration=False).needs_regeneration is False


# --------------------------------------------------------------------------- service / event stream
def test_event_stream_order_and_stored_answer_is_the_verified_one(chat_env, monkeypatch):
    monkeypatch.setattr(chat_module, "stream_chat", FakeDraft(GOOD + " An invented claim about 9999 deaths [S1]."))
    svc, me = ChatService(chat_env), owner_id_for(new_user_token())
    events, done = run(svc, me, "How many fatal accidents occurred in coal mines in 2019?")
    types = [e.type for e in events]
    assert types[0] == "status" and types[-1] == "done"
    assert types.index("token") < [i for i, e in enumerate(events) if e.type == "status"
                                   and e.data["stage"] == "verifying"][0] < types.index("citation") < types.index("done")
    assert done.data["status"] == PARTIAL and "9999" not in done.data["answer"]
    cite = next(e for e in events if e.type == "citation").data
    assert cite["citation"] == "sanket0404_2024.pdf, p. 159" and "stands at 51" in cite["excerpt"]
    stored = svc.store.messages(me, done.data["conversation_id"])
    assert [m["role"] for m in stored] == ["user", "assistant"]
    assert stored[1]["content"] == done.data["answer"] and "9999" in stored[1]["meta"]["draft"]
    assert stored[1]["meta"]["verification"]["removed"]


def test_regeneration_replaces_an_unsupported_draft(chat_env, monkeypatch):
    fake = FakeDraft("An invented accident at a mine [S1].", GOOD)
    monkeypatch.setattr(chat_module, "stream_chat", fake)
    svc, me = ChatService(chat_env), owner_id_for(new_user_token())
    events, done = run(svc, me, "How many fatal accidents occurred in coal mines in 2019?")
    assert any(e.type == "status" and e.data.get("reset_draft") for e in events)
    assert len(fake.calls) == 2 and "not support" in fake.calls[1][-1][1]
    assert done.data["status"] == VERIFIED and "51 fatal accidents" in done.data["answer"]


def test_follow_up_uses_conversation_and_memory_stays_context(chat_env, monkeypatch):
    fake = FakeDraft("A brick wall holding water failed at Sayal D in 2017 and two persons drowned [S1].")
    monkeypatch.setattr(chat_module, "stream_chat", fake)
    svc, me = ChatService(chat_env), owner_id_for(new_user_token())
    _, first = run(svc, me, "What happened at Sayal D colliery in 2017?")
    conv = first.data["conversation_id"]
    events, second = run(svc, me, "Which regulation did that finding cite?", conv)
    retrieved = next(e for e in events if e.type == "status" and e.data["stage"] == "retrieved")
    assert "Sayal" in retrieved.data["standalone_query"]  # resolved from the conversation
    assert fake.calls[-1][0] == ("user", "What happened at Sayal D colliery in 2017?")  # history passed
    # a new conversation on the same topic receives the earlier topic note, labelled as context only
    run(svc, me, "Earlier I asked about Sayal D colliery - remind me what failed there in 2017?")
    last_prompt = fake.calls[-1][-1][1]
    assert "context only - not evidence" in last_prompt and "Sayal" in last_prompt


def test_offline_mode_returns_passages_only(chat_env, monkeypatch):
    monkeypatch.setattr(service_module, "is_llm_enabled", lambda s=None: False)
    svc, me = ChatService(chat_env), owner_id_for(new_user_token())
    events, done = run(svc, me, "How many fatal accidents occurred in coal mines in 2019?")
    assert done.data["status"] == PASSAGES_ONLY and "No language model was used" in done.data["answer"]
    assert not any(e.type == "token" for e in events) and any(e.type == "citation" for e in events)


def test_answer_cache_is_reused_and_invalidated(chat_env, monkeypatch):
    fake = FakeDraft(GOOD)
    monkeypatch.setattr(chat_module, "stream_chat", fake)
    svc, me = ChatService(chat_env), owner_id_for(new_user_token())
    run(svc, me, "How many fatal accidents occurred in coal mines in 2019?")
    _, again = run(svc, me, "How many fatal accidents occurred in coal mines in 2019?")
    assert again.data["cached"] is True and len(fake.calls) == 1
    # simulate a changed PDF: a new document version must not see the old answer
    monkeypatch.setattr(ChatService, "doc_version", property(lambda self: "changed-version"))
    _, fresh = run(svc, me, "How many fatal accidents occurred in coal mines in 2019?")
    assert fresh.data["cached"] is False and len(fake.calls) == 2


def test_validation_rate_limit_and_foreign_conversations(chat_env, monkeypatch):
    monkeypatch.setattr(chat_module, "stream_chat", FakeDraft(GOOD))
    svc = ChatService(chat_env, limiter=RateLimiter(2))
    me, other = owner_id_for(new_user_token()), owner_id_for(new_user_token())
    assert run(svc, me, "   ")[0][0].type == "error"
    assert "limited" in run(svc, me, "x" * 5000)[0][0].data["message"]
    _, ok = run(svc, me, "How many fatal accidents occurred in coal mines in 2019?")
    assert run(svc, other, "hello there friend", ok.data["conversation_id"])[0][-1].data["message"] == \
        "Conversation not found."
    run(svc, me, "second question in the minute")
    assert "Too many questions" in run(svc, me, "third question")[0][0].data["message"]


def test_interrupted_generation_leaves_a_pending_marker(chat_env, monkeypatch):
    monkeypatch.setattr(chat_module, "stream_chat", FakeDraft(GOOD))
    svc, me = ChatService(chat_env), owner_id_for(new_user_token())
    stream = svc.ask(me, "How many fatal accidents occurred in coal mines in 2019?")
    for ev in stream:
        if ev.type == "token":
            break  # the user pressed Stop / closed the connection
    stream.close()
    conv = svc.store.list_conversations(me)[0]
    assert svc.store.messages(me, conv["id"])[-1]["status"] == PENDING


# --------------------------------------------------------------------------- HTTP API
@pytest.fixture()
def api_client(chat_env, monkeypatch):
    from fastapi.testclient import TestClient

    from src import api

    monkeypatch.setenv("SANKET_API_TOKEN", "test-server-token-123")
    monkeypatch.setattr(chat_module, "stream_chat", FakeDraft(GOOD))
    api.get_service.cache_clear()
    yield TestClient(api.app)
    api.get_service.cache_clear()


def _headers(user="user-token-aaaaaaaaaaaa", token="test-server-token-123"):
    return {"Authorization": f"Bearer {token}", "X-Sanket-User": user}


def test_api_requires_tokens(api_client, monkeypatch):
    assert api_client.get("/api/conversations").status_code == 401
    assert api_client.get("/api/conversations", headers=_headers(token="wrong")).status_code == 401
    assert api_client.get("/api/conversations", headers=_headers(user="short")).status_code == 400
    monkeypatch.setenv("SANKET_API_TOKEN", "")
    assert api_client.get("/api/conversations", headers=_headers()).status_code == 503


def test_api_streams_sse_events_and_isolates_users(api_client):
    r = api_client.post("/api/chat/stream", headers=_headers(),
                        json={"message": "How many fatal accidents occurred in coal mines in 2019?", "mode": "pdf"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = re.findall(r"^event: (\w+)$", r.text, re.M)
    assert events[0] == "status" and "token" in events and "citation" in events and events[-1] == "done"
    done = json.loads(r.text.strip().split("data: ")[-1])
    assert done["status"] == VERIFIED and done["conversation_id"]
    cid = done["conversation_id"]
    assert len(api_client.get(f"/api/conversations/{cid}/messages", headers=_headers()).json()) == 2
    intruder = _headers(user="another-user-bbbbbbbbbb")
    assert api_client.get(f"/api/conversations/{cid}/messages", headers=intruder).status_code == 404
    assert api_client.delete(f"/api/conversations/{cid}", headers=intruder).status_code == 404
    assert api_client.post("/api/chat/stream", headers=intruder,
                           json={"message": "hi there", "conversation_id": cid}).status_code == 404
    assert api_client.patch(f"/api/conversations/{cid}", headers=_headers(), json={"title": "Stats"}).status_code == 200
    assert api_client.get("/api/conversations", headers=_headers()).json()[0]["title"] == "Stats"
    assert api_client.put("/api/memory", headers=_headers(), json={"enabled": False}).json() == {"enabled": False}
    assert api_client.delete("/api/conversations", headers=_headers()).json()["deleted"] == 1
    assert api_client.post("/api/chat/stream", headers=_headers(), json={"message": ""}).status_code == 422


# --------------------------------------------------------------------------- floating UI
@pytest.fixture()
def ui_env(chat_env, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(chat_module, "stream_chat", FakeDraft(GOOD))
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def test_bot_page_answers_in_quick_mode_from_the_internet(ui_env, monkeypatch):
    from src.web_search import WebResult

    monkeypatch.setattr(service_module, "is_llm_enabled", lambda s=None: True)
    monkeypatch.setattr(service_module, "research", lambda q, t, **k: ([WebResult(
        "Coal mine safety 2019 - PIB", "https://pib.gov.in/coal2019",
        "In 2019 coal mines recorded 51 fatal accidents with 56 fatalities, according to DGMS data.", "2020-02-01")], []))

    def fake_stream(system, messages, settings=None, used=None):
        assert "internet search results" in system and "<web_results" in messages[-1][1]
        if used is not None:
            used.update(provider="fake", model="fake-model", skipped=[])
        yield "Coal mines recorded 51 fatal accidents with 56 fatalities in 2019 [W1]. "
        yield "Another figure of 9999 deaths appears nowhere [W1]."

    monkeypatch.setattr(service_module, "stream_chat", fake_stream)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.switch_page("pages/0_Bot.py")
    at.run()
    assert not at.exception and len(at.chat_input) == 1
    assert any("Where should we begin?" in m.value for m in at.markdown)
    assert any(b.key == "suggest_0" for b in at.button)  # suggestion buttons on the start screen
    # chat-only sidebar: no app title or page list, just the chat tools
    assert not any(m.value == "**Inundation**" for m in at.sidebar.markdown)
    assert at.button(key="bot_new").label == "New chat"
    # answer-mode dropdown next to the message box, with the three modes
    assert at.selectbox(key="bot_mode").options == ["Research report", "Quick answer", "PDF answer"]
    assert at.selectbox(key="bot_mode").value == "research"
    at.selectbox(key="bot_mode").set_value("quick").run()
    at.chat_input(key="bot_input").set_value("How many fatal accidents occurred in coal mines in 2019?").run()
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    answer = next(m.value for m in at.markdown if "51 fatal accidents" in m.value)
    assert "9999" not in answer and "Citations checked" in text
    assert "Removed:" in text  # the removed sentence is listed openly under "How this answer was checked"
    assert "Answered by" not in text + "\n".join(c.value for c in at.caption)
    assert any(e.label == "Sources: 1 from the internet" for e in at.expander)
    # the conversation appears in Recents and can be reopened after starting a new chat
    conv_buttons = [b for b in at.button if b.key and b.key.startswith("open_")]
    assert len(conv_buttons) == 1
    at.button(key="bot_new").click().run()
    assert not any("51 fatal accidents" in m.value for m in at.markdown)
    at.button(key=conv_buttons[0].key).click().run()
    assert any("51 fatal accidents" in m.value for m in at.markdown)

# --------------------------------------------------------------------------- fast mode and web facts
def test_fast_mode_checks_citations_and_numbers_without_a_model_call(chat_env, monkeypatch):
    monkeypatch.setenv("SANKET_CHAT_VERIFY_MODE", "fast")

    def no_model(*a, **k):
        raise AssertionError("fast mode must not call the checker model")

    monkeypatch.setattr(verification_module, "complete_text", no_model)
    passages = [Passage("S1", "page", "vol.pdf", [159], "fatal accidents in 2019 stands at 51 with 56 fatalities")]
    result = verify_answer("In 2019 there were 51 fatal accidents and 56 fatalities [S1]. "
                           "A further 9999 deaths were recorded [S1]. Unrelated claim cites a missing source [S5].",
                           passages, get_settings())
    assert result.status == "citations_checked" and not result.checker_used
    assert "51 fatal accidents" in result.answer and "9999" not in result.answer and "[S5]" not in result.answer


def test_web_facts_are_capped_at_twenty_percent_and_cited(chat_env, monkeypatch):
    monkeypatch.setenv("SANKET_CHAT_VERIFY_MODE", "fast")
    passages = [
        Passage("S1", "page", "vol.pdf", [304], "27-12-1975 Chasnalla 375 killed Irruption of Water"),
        Passage("W1", "web", "Chasnala mining disaster - Wikipedia", [], "The disaster happened near Dhanbad when "
                "water from an adjacent reservoir flooded the mine.", url="https://en.wikipedia.org/wiki/x"),
    ]
    draft = ("The SANKET coal volume records an irruption of water at Chasnalla on 27-12-1975 in which 375 persons "
             "were killed, the deadliest water-related accident in the table of major accidents [S1]. "
             "Wikipedia says an adjacent reservoir flooded it [W1]. "
             "Wikipedia also says rescue pumps had to be brought from abroad and took weeks to arrive [W1].")
    result = verify_answer(draft, passages, get_settings())
    assert result.general_share <= 0.2 + 1e-9
    assert "[W1]" in result.answer and "[S1]" in result.answer
    assert any("above the 20% share" in c.removal_reason for c in result.removed)


def test_when_pdfs_lack_the_answer_web_facts_are_labelled(chat_env, monkeypatch):
    monkeypatch.setenv("SANKET_CHAT_VERIFY_MODE", "fast")
    passages = [Passage("W1", "web", "Example", [], "The mine reopened in 1980.", url="https://example.org/a")]
    result = verify_answer("The mine reopened in 1980 according to Example.org [W1].", passages, get_settings())
    assert result.status == INSUFFICIENT
    assert result.answer.startswith("This cannot be established from the two SANKET volumes provided. "
                                    "From internet sources")


def test_service_runs_web_search_and_returns_web_citations(chat_env, monkeypatch):
    from src.web_search import WebResult

    monkeypatch.setenv("SANKET_CHAT_VERIFY_MODE", "fast")
    monkeypatch.setenv("SANKET_CHAT_WEB", "true")
    searched = []

    def fake_search(query, n, timeout):
        searched.append(query)
        return [WebResult("DGMS - Wikipedia", "https://en.wikipedia.org/wiki/DGMS",
                          "The Directorate General of Mines Safety is a regulatory agency under the Ministry of "
                          "Labour and Employment.")], None

    monkeypatch.setattr(service_module, "search_web", fake_search)
    draft = (GOOD + " The same page states that 6 persons were seriously injured in these fatal accidents in "
             "2019 [S1]. DGMS is a regulatory agency [W1].")
    fake = FakeDraft(draft)
    monkeypatch.setattr(chat_module, "stream_chat", fake)
    svc, me = ChatService(get_settings()), owner_id_for(new_user_token())
    events, done = run(svc, me, "How many fatal accidents occurred in coal mines in 2019?")
    assert searched and done.data["status"] == "citations_checked"
    assert "<web_results" in fake.calls[0][-1][1] and 'id="W1"' in fake.calls[0][-1][1]
    web = [e.data for e in events if e.type == "citation" and e.data["kind"] == "web"]
    assert web and web[0]["url"] == "https://en.wikipedia.org/wiki/DGMS"
    assert not any(e.type == "status" and e.data.get("stage") == "regenerating" for e in events)


def test_web_search_failure_falls_back_to_pdfs(chat_env, monkeypatch):
    monkeypatch.setenv("SANKET_CHAT_VERIFY_MODE", "fast")
    monkeypatch.setenv("SANKET_CHAT_WEB", "true")
    monkeypatch.setattr(service_module, "search_web", lambda q, n, t: ([], "Web search failed (Timeout)."))
    monkeypatch.setattr(chat_module, "stream_chat", FakeDraft(GOOD))
    svc, me = ChatService(get_settings()), owner_id_for(new_user_token())
    _, done = run(svc, me, "How many fatal accidents occurred in coal mines in 2019?")
    assert done.data["status"] == "citations_checked"
    stored = svc.store.messages(me, done.data["conversation_id"])[-1]
    assert stored["meta"]["web_error"] == "Web search failed (Timeout)."


# --------------------------------------------------------------------------- research mode
def test_operator_matching_is_exact():
    from src.research import Understanding, match_operator

    ops = ["Eastern Coalfields Ltd.", "South Eastern Coalfields Ltd.", "Central Coalfields Ltd."]
    assert match_operator(Understanding(organization_abbreviation="ECL"), ops) == ["Eastern Coalfields Ltd."]
    assert match_operator(Understanding(organization_full_name="South Eastern Coalfields Limited"), ops) == \
        ["South Eastern Coalfields Ltd."]
    assert match_operator(Understanding(), ops) == []


def test_research_pipeline_end_to_end(chat_env, monkeypatch):
    from src import llm as llm_module
    from src import research as research_module
    from src.web_search import WebResult

    def fake_complete(system, messages, settings=None, json_mode=False, used=None):
        if "interpret questions" in system:
            return json.dumps({"restated_question": "Inundation conditions at ECL", "user_context": {"age": "50",
                               "role": "CMD", "role_expanded": "Chairman-cum-Managing Director"},
                               "organization_abbreviation": "ECL", "organization_full_name": "Eastern Coalfields Limited",
                               "locations": ["West Bengal"], "mines": [], "subject": "inundation conditions",
                               "intent": "assess hazards", "explicit_facts": ["age 50", "CMD at ECL"],
                               "assumptions": ["ECL means Eastern Coalfields Limited"], "needs_clarification": False,
                               "clarification_question": None, "scope": "ECL mines", "web_queries": ["ECL inundation"],
                               "pdf_queries": ["Eastern Coalfields drowning"]})
        assert "fact-check" in system
        return json.dumps({"claims": [
            {"claim": "A drowning occurred at Kuardih colliery on 25 July 2017.", "web": ["W1"], "verdict": "supported",
             "pdf": ["S1"], "note": "Matches the PDF record."},
            {"claim": "ECL ran monsoon control rooms in 2024.", "web": ["W2"], "verdict": "supported", "pdf": ["S99"],
             "note": "invented PDF id"},
            {"claim": "Unsourced claim with no web id at all.", "web": [], "verdict": "not_addressed", "pdf": [],
             "note": ""}]})

    def fake_research(queries, terms, **kw):
        return [WebResult("Kuardih drowning - news", "https://news.example.in/kuardih", "Kuardih colliery of Eastern "
                          "Coalfields: a worker drowned on 25 July 2017 after an inrush of water.", "2017-07-26"),
                WebResult("ECL safety bulletin", "https://easterncoalfields.in/safety.pdf", "Eastern Coalfields "
                          "monsoon control rooms operate from 15 June to 15 October 2024.", "2024-06-10")], []

    report = ("## Executive summary\nA worker drowned at Kuardih colliery on 25 July 2017 [W1], which the SANKET "
              "records confirm [S1]. ECL ran monsoon control rooms in 2024 [W2]. An invented source [W9].")
    stream_calls = []

    def fake_stream(system, messages, settings=None, used=None):
        stream_calls.append(messages)
        if used is not None:
            used.update(provider="fake", model="fake-model", skipped=[])
        yield report

    monkeypatch.setattr(research_module, "complete_text", fake_complete)
    monkeypatch.setattr(research_module, "research", fake_research)
    monkeypatch.setattr(llm_module, "stream_chat", fake_stream)
    svc, me = ChatService(get_settings()), owner_id_for(new_user_token())
    events = list(svc.ask(me, "I am a 50 year old, placed as CMD at ECL, what are the inundation conditions there?",
                          mode="research"))
    stages = [e.data["stage"] for e in events if e.type == "status"]
    for stage in ("understanding", "understood", "researching", "web_done", "pdf_done", "fact_checking", "writing",
                  "verifying"):
        assert stage in stages
    done = events[-1].data
    assert events[-1].type == "done" and done["status"] == "research_report"
    # owner match is exact: SECL mines are not part of ECL's records
    stored = svc.store.messages(me, done["conversation_id"])[-1]["meta"]
    assert stored["owners"] == ["Eastern Coalfields Ltd."]
    # fact-check rows are validated: invented PDF ids lose their verdict, rows without a web source are dropped
    checks = done["fact_check"]
    assert [c["verdict"] for c in checks] == ["supported", "not_addressed"]
    assert checks[0]["pdf_pages"] == ["SANKET records for Eastern Coalfields Ltd."]
    # charts come from PDF data for ECL only
    table = next(c for c in done["charts"] if c["kind"] == "table")
    mines = {row[1] for row in table["rows"]}
    assert "KUARDIH" in mines and "GEVRA PROJECT" not in mines and "KUSMUNDA OCP" not in mines
    # invented citation marker removed; real web sources cited with their links
    assert "[W9]" not in done["answer"] and "[W1]" in done["answer"]
    web = [e.data for e in events if e.type == "citation" and e.data["kind"] == "web"]
    assert {w["url"] for w in web} == {"https://news.example.in/kuardih", "https://easterncoalfields.in/safety.pdf"}
    # the report prompt carried web evidence first, the PDF passages and the fact-check
    prompt = stream_calls[0][-1][1]
    assert prompt.index("<web_sources") < prompt.index("<pdf_passages>") < prompt.index("<pdf_fact_check>")


def test_research_asks_for_clarification_when_nothing_can_be_scoped(chat_env, monkeypatch):
    from src import research as research_module

    monkeypatch.setattr(research_module, "complete_text", lambda *a, **k: json.dumps({
        "needs_clarification": True, "clarification_question": "Which mine or area do you mean?",
        "web_queries": [], "pdf_queries": []}))
    svc, me = ChatService(get_settings()), owner_id_for(new_user_token())
    done = list(svc.ask(me, "what about there?", mode="research"))[-1]
    assert done.data["status"] == "clarification" and done.data["answer"] == "Which mine or area do you mean?"


def test_quick_mode_uses_only_internet_sources(chat_env, monkeypatch):
    from src.web_search import WebResult

    monkeypatch.setattr(service_module, "research", lambda q, t, **k: ([WebResult(
        "ECL news", "https://news.example.in/ecl", "Eastern Coalfields reported water seepage at a mine in 2024.",
        "2024-08-01")], []))
    prompts = []

    def fake_stream(system, messages, settings=None, used=None):
        prompts.append(messages[-1][1])
        yield "Eastern Coalfields reported water seepage at a mine in 2024 [W1]. A fake source [S3]."

    monkeypatch.setattr(service_module, "stream_chat", fake_stream)
    svc, me = ChatService(get_settings()), owner_id_for(new_user_token())
    events = list(svc.ask(me, "Is there water seepage at ECL mines?", mode="quick"))
    done = events[-1].data
    assert done["status"] == "citations_checked" and "[W1]" in done["answer"] and "[S3]" not in done["answer"]
    assert "SANKET" not in done["answer"] and "<pdf" not in prompts[0] and "<passages>" not in prompts[0]
    cites = [e.data for e in events if e.type == "citation"]
    assert [c["kind"] for c in cites] == ["web"] and cites[0]["url"] == "https://news.example.in/ecl"


def test_quick_mode_with_no_internet_results(chat_env, monkeypatch):
    monkeypatch.setattr(service_module, "research", lambda q, t, **k: ([], ["Web search failed (Timeout)."]))
    svc, me = ChatService(get_settings()), owner_id_for(new_user_token())
    done = list(svc.ask(me, "anything at all here?", mode="quick"))[-1].data
    assert done["status"] == "insufficient_evidence" and "could not find any internet sources" in done["answer"]
