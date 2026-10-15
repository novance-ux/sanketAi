"""Chat orchestration: one question in, a stream of typed events out.

Used by the floating chat panel, the full chat page and the HTTP API (src/api.py), so every
surface follows the same pipeline:

    validate + rate-limit -> store question -> resolve follow-up into a standalone query
    -> retrieve passages from both PDFs (cached per document version)
    -> stream a DRAFT (event "token", provisional) -> verify claims against the passages
    -> regenerate once with feedback if too little survives -> emit citations
    -> store the verified answer -> event "done"

Event types:
    status    {"stage": ..., "message": ..., ...}    progress (retrieving, generating, verifying, ...)
    token     {"text": ...}                          provisional draft text - NOT verified
    citation  {"sid", "source_filename", "pages", "citation", "excerpt", "kind"}  sources of the final answer
    done      {"status", "label", "answer", "conversation_id", "message_id", "verification", ...}
    error     {"message", "recoverable"}

"done" is the only event that carries the answer to show as final, and its status says honestly
whether verification ran and what it found.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Iterator

from .chat import ChatTurn, DocumentChat, Passage
from .chat_cache import make_key, open_cache
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
    ChatStore,
    NotFound,
)
from .config import Settings, get_settings
from .database import DatabaseNotReady
from .llm import LLMUnavailable, complete_text, is_llm_enabled
from .verification import VerificationResult, verify_answer
from .web_search import search_web

log = logging.getLogger(__name__)
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sanket-web")

_FOLLOW_UP = re.compile(
    r"\b(it|its|that|this|these|those|they|them|there|he|she|previous|above|earlier|same|further|more|"
    r"elaborate|explain|again|also|what about|and the)\b", re.I)

# Bump when retrieval or verification logic changes so cached results from older logic are not reused.
RETRIEVAL_VERSION = 6

# The user refers back to earlier conversations: only then are long-term memory notes consulted.
_PAST_REFERENCE = re.compile(
    r"\b(earlier|before|previous(?:ly)?|last time|we (?:discussed|talked)|you (?:said|told|mentioned)|"
    r"remind|other conversation|past conversation|i asked)\b", re.I)

REWRITE_PROMPT = """Rewrite the user's latest message as one standalone search query for two DGMS mine-accident
statistics volumes, resolving references such as "that mine", "the previous answer" or "explain it further"
from the conversation. Keep mine names, years and technical terms. Output only the query, on one line,
with no quotes or commentary. If the message is already standalone, repeat it."""


@dataclass
class ChatEvent:
    type: str
    data: dict

    def sse(self) -> str:
        return f"event: {self.type}\ndata: {json.dumps(self.data, ensure_ascii=False, default=str)}\n\n"


class RateLimiter:
    """Sliding-window limit per owner (in-process; protects the model quota from abuse)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, owner_id: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            q = self._hits[owner_id]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


def _passage_to_dict(p: Passage) -> dict:
    return asdict(p)


def _passage_from_dict(d: dict) -> Passage:
    return Passage(**d)


class ChatService:
    def __init__(self, settings: Settings | None = None, doc_chat: DocumentChat | None = None,
                 store: ChatStore | None = None, cache=None, limiter: RateLimiter | None = None):
        self.settings = settings or get_settings()
        self.doc_chat = doc_chat or DocumentChat.from_settings(self.settings)
        self.store = store or ChatStore(self.settings.chat_db_path)
        self._cache = cache
        self.limiter = limiter or RateLimiter(self.settings.chat_rate_limit_per_min)

    # ------------------------------------------------------------------ helpers
    @property
    def doc_version(self) -> str:
        meta = self.doc_chat.db.meta()
        return f"{meta.get('fingerprint', 'unknown')}:{meta.get('pipeline_version', '')}:r{RETRIEVAL_VERSION}"

    @property
    def cache(self):
        if self._cache is None or getattr(self._cache, "doc_version", None) != self.doc_version:
            self._cache = open_cache(self.settings.redis_url, self.settings.cache_path, self.doc_version)
        return self._cache

    def _model_chain(self) -> list[str]:
        return [f"{c.llm_provider}:{c.chat_model()}" for c in self.settings.provider_chain()]

    def _history(self, owner_id: str, conversation_id: str) -> list[ChatTurn]:
        turns: list[ChatTurn] = []
        question = None
        for m in self.store.messages(owner_id, conversation_id):
            if m["role"] == "user":
                question = m["content"]
            elif question is not None and m["status"] in (VERIFIED, PARTIAL, UNVERIFIED, INSUFFICIENT, CHECKED,
                                                          NOT_CHECKED):
                turns.append(ChatTurn(question, m["content"], [], None, "llm"))
                question = None
        return turns

    def standalone_query(self, question: str, history: list[ChatTurn]) -> str:
        if not history or not (len(question.split()) < 12 or _FOLLOW_UP.search(question)):
            return question
        convo = []
        for t in history[-3:]:
            convo.append(f"User: {t.question}")
            convo.append(f"Assistant: {t.answer[:600]}")
        convo.append(f"User (latest): {question}")
        try:
            text = complete_text(REWRITE_PROMPT, [("user", "\n".join(convo))], self.settings)
            line = text.strip().splitlines()[0].strip().strip('"').strip()
            if 3 <= len(line) <= 400:
                return line
        except (LLMUnavailable, IndexError):
            pass
        return f"{history[-1].question} {question}"

    def _retrieve(self, query: str, category: str | None, k: int) -> list[Passage]:
        key = make_key("retrieval", self.doc_version, query.lower().strip(), category, k)
        cached = self.cache.get(key)
        if cached is not None:
            return [_passage_from_dict(d) for d in cached]
        passages = self.doc_chat.retrieve(query, category=category, limit=k)
        self.cache.set(key, [_passage_to_dict(p) for p in passages], self.settings.cache_ttl_s)
        return passages

    def _web(self, query: str) -> tuple[list[Passage], str | None]:
        """Internet results as W-passages (cached for a day; only the query is sent)."""
        key = make_key("web", "web", query.lower().strip(), self.settings.chat_web_results)
        cached = self.cache.get(key)
        if cached is not None:
            return [_passage_from_dict(d) for d in cached], None
        results, error = search_web(query, self.settings.chat_web_results, self.settings.chat_web_timeout_s)
        passages = [Passage(sid=f"W{n}", kind="web", source_filename=r.title, pages=[], text=r.snippet,
                            matched_by=["web search"], chunk_id=r.url, url=r.url)
                    for n, r in enumerate(results, 1)]
        if passages:
            self.cache.set(key, [_passage_to_dict(p) for p in passages], self.settings.cache_ttl_s)
        return passages, error

    def _passages_still_indexed(self, passages: list[Passage]) -> bool:
        pages = {p.chunk_id for p in passages if p.kind == "page"}
        records = {p.chunk_id for p in passages if p.kind == "record"}
        if any(p.kind == "index" for p in passages) and not self.doc_chat.water_index_passage():
            return False
        if pages and len(self.doc_chat.db.get_page_chunks(pages)) != len(pages):
            return False
        for rid in records:
            if not rid or not self.doc_chat.db.get_mine_for_incident(rid):
                return False
        return True

    @staticmethod
    def citation_events(answer: str, passages: list[Passage]) -> list[ChatEvent]:
        used = []
        for m in re.finditer(r"\[\s*([SW]\d+(?:\s*[,;]\s*[SW]\d+)*)\s*\]", answer):
            for sid in re.split(r"[,;]", m.group(1)):
                if sid.strip() not in used:
                    used.append(sid.strip())
        by_id = {p.sid: p for p in passages}
        return [ChatEvent("citation", {
            "sid": sid, "source_filename": by_id[sid].source_filename, "pages": by_id[sid].pages,
            "citation": by_id[sid].citation, "kind": by_id[sid].kind, "mine": by_id[sid].mine_name,
            "url": by_id[sid].url, "excerpt": by_id[sid].text[:3000]}) for sid in used if sid in by_id]

    # ------------------------------------------------------------------ main entry
    def ask(self, owner_id: str, message: str, conversation_id: str | None = None,
            category: str | None = None) -> Iterator[ChatEvent]:
        message = " ".join((message or "").split())
        if not message:
            yield ChatEvent("error", {"message": "Please type a question.", "recoverable": True})
            return
        if len(message) > self.settings.chat_max_message_chars:
            yield ChatEvent("error", {"message": f"Questions are limited to {self.settings.chat_max_message_chars} "
                                                 "characters.", "recoverable": True})
            return
        if not self.limiter.allow(owner_id):
            yield ChatEvent("error", {"message": "Too many questions in the last minute. Please wait a moment.",
                                      "recoverable": True})
            return
        try:
            if conversation_id:
                self.store.get_conversation(owner_id, conversation_id)
            else:
                conversation_id = self.store.create_conversation(owner_id)["id"]
        except NotFound:
            yield ChatEvent("error", {"message": "Conversation not found.", "recoverable": False})
            return
        history = self._history(owner_id, conversation_id)
        self.store.add_message(owner_id, conversation_id, "user", message, "received")
        self.store.autotitle(owner_id, conversation_id, message)
        reply_id = self.store.add_message(owner_id, conversation_id, "assistant", "", PENDING)
        yield ChatEvent("status", {"stage": "started", "conversation_id": conversation_id, "message_id": reply_id,
                                   "message": "Question received"})
        try:
            yield from self._answer(owner_id, conversation_id, reply_id, message, history, category)
        except DatabaseNotReady as exc:
            self.store.update_message(owner_id, reply_id, content=str(exc), status=ERROR, meta={})
            yield ChatEvent("error", {"message": "The document index has not been built. Run "
                                                 "`python -m src.ingestion`.", "recoverable": False})
        except Exception as exc:  # never leave the UI without a terminal event
            log.exception("Chat pipeline failed")
            self.store.update_message(owner_id, reply_id, content="The answer could not be produced.",
                                      status=ERROR, meta={"error": type(exc).__name__})
            yield ChatEvent("error", {"message": f"Something went wrong while answering ({type(exc).__name__}). "
                                                 "Please try again.", "recoverable": True})
        finally:
            try:
                self.store.refresh_summary(owner_id, conversation_id)
            except NotFound:
                pass

    def _answer(self, owner_id, conversation_id, reply_id, message, history, category) -> Iterator[ChatEvent]:
        s = self.settings
        llm_ready = is_llm_enabled(s)
        if history and llm_ready:
            yield ChatEvent("status", {"stage": "understanding", "message": "Interpreting your follow-up question"})
        query = self.standalone_query(message, history) if llm_ready else (
            f"{history[-1].question} {message}" if history and len(message.split()) < 6 else message)

        use_web = s.chat_web and llm_ready
        yield ChatEvent("status", {"stage": "retrieving",
                                   "message": "Searching both PDFs" + (" and the web" if use_web else "")})
        web_future = _POOL.submit(self._web, query) if use_web else None
        passages = self._retrieve(query, category, s.chat_passages)
        web_passages, web_error = web_future.result() if web_future else ([], None)
        volumes = sorted({p.source_filename for p in passages})
        yield ChatEvent("status", {"stage": "retrieved",
                                   "message": f"{len(passages)} PDF passages" + (
                                       f", {len(web_passages)} web results" if use_web else "") + " found",
                                   "passages": len(passages), "web_results": len(web_passages),
                                   "volumes": volumes, "standalone_query": query})

        memories = ([m["summary"] for m in self.store.relevant_memories(owner_id, query, conversation_id)]
                    if _PAST_REFERENCE.search(message) else [])

        base_meta = {"question": message, "standalone_query": query, "category": category,
                     "doc_version": self.doc_version, "memories_used": len(memories),
                     "web_results": len(web_passages), "web_error": web_error,
                     "check_mode": s.chat_verify_mode}

        if not passages:
            answer = ("This cannot be established from the two SANKET volumes provided: no passage in either "
                      "volume matches the question.")
            yield from self._finish(owner_id, conversation_id, reply_id, answer, INSUFFICIENT, [], None, base_meta)
            return

        if not llm_ready:
            answer = self.doc_chat.extractive_answer(passages, "no language model is configured")
            yield from self._finish(owner_id, conversation_id, reply_id, answer, PASSAGES_ONLY, passages, None,
                                    base_meta)
            return

        cache_key = None
        if not history:
            cache_key = make_key("answer", self.doc_version, query.lower().strip(), category, self._model_chain(),
                                 s.chat_max_general_share, s.chat_verify_mode, use_web, memories)
            hit = self.cache.get(cache_key)
            if hit:
                cached_passages = [_passage_from_dict(d) for d in hit["passages"]]
                if self._passages_still_indexed(cached_passages):
                    yield ChatEvent("status", {"stage": "cache", "message": "Reusing an earlier checked answer for "
                                                                           "this document version"})
                    meta = {**base_meta, **hit["meta"], "cached": True}
                    yield from self._finish(owner_id, conversation_id, reply_id, hit["answer"], hit["status"],
                                            cached_passages, None, meta)
                    return
                self.cache.delete(cache_key)

        feedback = None
        result: VerificationResult | None = None
        used: dict = {}
        attempts = 1 + (max(0, s.chat_max_regenerations) if s.chat_verify_mode == "thorough" else 0)
        for attempt in range(attempts):
            if attempt:
                yield ChatEvent("status", {"stage": "regenerating", "reset_draft": True,
                                           "message": "Some statements were not supported; retrying with more "
                                                      "passages"})
                passages = self._retrieve(f"{query} {message}" if query != message else query, category,
                                          int(s.chat_passages * 1.5) + 1)
                pdf_passages = passages
            if not attempt:
                pdf_passages = passages
            passages = pdf_passages + web_passages
            yield ChatEvent("status", {"stage": "generating", "message": "Writing the answer"})
            draft_parts: list[str] = []
            try:
                for piece in self.doc_chat.stream_answer(message if query == message else
                                                         f"{message}\n(Interpreted as: {query})",
                                                         passages, history, used=used, memories=memories,
                                                         feedback=feedback):
                    draft_parts.append(piece)
                    yield ChatEvent("token", {"text": piece})
            except LLMUnavailable as exc:
                answer = self.doc_chat.extractive_answer(passages, str(exc))
                yield ChatEvent("status", {"stage": "model_unavailable", "message": str(exc)})
                yield from self._finish(owner_id, conversation_id, reply_id, answer, PASSAGES_ONLY, passages, None,
                                        {**base_meta, "error": str(exc)})
                return
            draft = "".join(draft_parts)
            yield ChatEvent("status", {"stage": "verifying", "message": (
                "Checking each statement against the sources" if s.chat_verify_mode == "thorough"
                else "Checking citations")})
            result = verify_answer(draft, passages, s, allow_regeneration=attempt + 1 < attempts)
            if not result.needs_regeneration:
                break
            feedback = ("Your previous draft contained statements that the passages do not support:\n"
                        + "\n".join(f"- {c.text[:200]} ({c.removal_reason})" for c in result.removed[:8])
                        + "\nWrite the answer again using only what the passages state. If they do not answer "
                          "the question, say so.")

        meta = {**base_meta, "verification": result.summary(), "draft": draft,
                "answered_by": f"{used.get('provider')} / {used.get('model')}" if used else None,
                "fallback": used.get("skipped") if used else None, "cached": False}
        events = list(self._finish(owner_id, conversation_id, reply_id, result.answer, result.status, passages,
                                   result, meta))
        if cache_key and result.status in (VERIFIED, PARTIAL):
            self.cache.set(cache_key, {"answer": result.answer, "status": result.status,
                                       "passages": [_passage_to_dict(p) for p in passages],
                                       "meta": {k: v for k, v in meta.items() if k not in ("cached",)}},
                           s.cache_ttl_s)
        yield from events

    def _finish(self, owner_id, conversation_id, reply_id, answer, status, passages, result, meta) -> Iterator[ChatEvent]:
        citations = self.citation_events(answer, passages)
        if status == PASSAGES_ONLY:
            citations = [ChatEvent("citation", {"sid": p.sid, "source_filename": p.source_filename, "pages": p.pages,
                                                "citation": p.citation, "kind": p.kind, "mine": p.mine_name,
                                                "excerpt": p.text[:3000]}) for p in passages[:4]]
        yield from citations
        meta = {**meta, "citations": [c.data for c in citations]}
        self.store.update_message(owner_id, reply_id, content=answer, status=status, meta=meta)
        yield ChatEvent("done", {
            "status": status, "label": STATUS_LABELS[status], "answer": answer,
            "conversation_id": conversation_id, "message_id": reply_id,
            "verification": result.summary() if result else meta.get("verification"),
            "answered_by": meta.get("answered_by"), "cached": meta.get("cached", False),
        })
