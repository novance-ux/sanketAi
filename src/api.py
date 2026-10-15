"""HTTP API for the SANKET chat (FastAPI, Server-Sent Events).

Run:  uvicorn src.api:app --host 127.0.0.1 --port 8600

Authentication (both required):
* ``Authorization: Bearer <SANKET_API_TOKEN>`` - server secret from .env. Without it configured the
  API refuses every request (503), so it is never open by accident.
* ``X-Sanket-User: <user token>`` - an unguessable per-user token (16+ characters). Conversations are
  owned by its SHA-256; one user can never read or change another user's conversations.

Streaming contract - POST /api/chat/stream returns ``text/event-stream`` with events:
    status   {"stage", "message", ...}       progress
    token    {"text"}                        provisional draft text, NOT verified
    citation {"sid", "source_filename", "pages", "citation", "excerpt", "kind", "mine"}
    done     {"status", "label", "answer", "conversation_id", "message_id", "verification", ...}
    error    {"message", "recoverable"}
Only "done" carries the answer to display as final; its "status" states whether verification succeeded.
"""
from __future__ import annotations

import hmac
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .chat_service import ChatService
from .chat_store import NotFound, owner_id_for
from .config import get_settings

app = FastAPI(title="Inundation chat API", version="1.0.0",
              description="PDF-grounded, claim-verified question answering over the two DGMS SANKET volumes.")


@lru_cache(maxsize=1)
def get_service() -> ChatService:
    return ChatService(get_settings())


def owner(authorization: str = Header(default=""), x_sanket_user: str = Header(default="")) -> str:
    expected = get_settings().api_token
    if not expected:
        raise HTTPException(503, "The API is disabled. Set SANKET_API_TOKEN in .env to enable it.")
    supplied = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(401, "Invalid or missing API token.")
    try:
        return owner_id_for(x_sanket_user)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    category: Literal["coal", "metal_non_coal"] | None = None
    # quick = internet only; research = web research + PDF fact-check report; pdf = PDFs first
    mode: Literal["quick", "research", "pdf"] = "quick"


class TitleRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class MemoryRequest(BaseModel):
    enabled: bool


@app.get("/api/health")
def health() -> dict:
    s = get_settings()
    return {"ok": True, "api_enabled": bool(s.api_token),
            "providers": [c.llm_provider for c in s.provider_chain()]}


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest, owner_id: str = Depends(owner)) -> StreamingResponse:
    service = get_service()
    if req.conversation_id:
        try:
            service.store.get_conversation(owner_id, req.conversation_id)
        except NotFound as exc:
            raise HTTPException(404, "Conversation not found.") from exc

    def events():
        for ev in service.ask(owner_id, req.message, req.conversation_id, req.category, req.mode):
            yield ev.sse()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/conversations")
def list_conversations(owner_id: str = Depends(owner)) -> list[dict]:
    return [{k: c[k] for k in ("id", "title", "created_at", "updated_at", "message_count")}
            for c in get_service().store.list_conversations(owner_id)]


@app.post("/api/conversations", status_code=201)
def create_conversation(req: TitleRequest | None = None, owner_id: str = Depends(owner)) -> dict:
    c = get_service().store.create_conversation(owner_id, req.title if req else "New conversation")
    return {k: c[k] for k in ("id", "title", "created_at", "updated_at")}


@app.get("/api/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str, owner_id: str = Depends(owner)) -> list[dict]:
    try:
        return [{k: m[k] for k in ("id", "role", "content", "status", "meta", "created_at")}
                for m in get_service().store.messages(owner_id, conversation_id)]
    except NotFound as exc:
        raise HTTPException(404, "Conversation not found.") from exc


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, req: TitleRequest, owner_id: str = Depends(owner)) -> dict:
    try:
        get_service().store.rename_conversation(owner_id, conversation_id, req.title)
    except NotFound as exc:
        raise HTTPException(404, "Conversation not found.") from exc
    return {"ok": True}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, owner_id: str = Depends(owner)) -> dict:
    try:
        get_service().store.delete_conversation(owner_id, conversation_id)
    except NotFound as exc:
        raise HTTPException(404, "Conversation not found.") from exc
    return {"ok": True}


@app.delete("/api/conversations")
def clear_history(owner_id: str = Depends(owner)) -> dict:
    return {"deleted": get_service().store.clear_history(owner_id)}


@app.get("/api/memory")
def get_memory(owner_id: str = Depends(owner)) -> dict:
    store = get_service().store
    return {"enabled": store.memory_enabled(owner_id), "notes": store.memories(owner_id)}


@app.put("/api/memory")
def set_memory(req: MemoryRequest, owner_id: str = Depends(owner)) -> dict:
    get_service().store.set_memory_enabled(owner_id, req.enabled)
    return {"enabled": req.enabled}


@app.delete("/api/memory")
def delete_memory(owner_id: str = Depends(owner)) -> dict:
    get_service().store.delete_memories(owner_id)
    return {"ok": True}
