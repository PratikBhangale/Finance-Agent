"""FastAPI server exposing the equity research ADK agent via a streaming
Server-Sent-Events (SSE) endpoint, plus REST routes for managing chat
sessions (list / create / fetch / delete) so a UI can render a sidebar of
past conversations, each with an auto-generated title.

Chat streaming endpoint events, from `/chat/stream`:
  - "session"      : first event; carries the session_id for this turn.
  - "thought"      : the model's internal reasoning (requires the agent to be
                      configured with `ThinkingConfig(include_thoughts=True)`).
  - "tool_call"    : a request from the model to invoke one of the finance
                      tools, with its name and arguments.
  - "tool_result"  : the result returned by a tool after execution.
  - "text"         : incremental (partial) or aggregated model output text.
  - "final"        : the final, complete response text for the turn.
  - "session_title": emitted once for a brand-new session, after an
                      auto-generated title has been produced and persisted.
  - "error"        : any error raised while running the agent.
  - "done"         : sentinel marking the end of the stream for this request.

Session management endpoints:
  - POST   /sessions                       Create a new (empty) session.
  - GET    /sessions?user_id=...           List session summaries (sidebar).
  - GET    /sessions/{session_id}          Full chat history for a session.
  - DELETE /sessions/{session_id}          Delete a session.
  - GET    /sessions/{session_id}/title    Poll for the async-generated title.

Run the server with:

    uv run uvicorn main:app --reload

Then stream a response with:

    curl -N -X POST http://127.0.0.1:8000/chat/stream \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"Give me a quick analysis of AAPL\", \"user_id\": \"demo\"}"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from dotenv import load_dotenv

# Load environment variables (e.g. GOOGLE_CLOUD_PROJECT, GOOGLE_GENAI_USE_ENTERPRISE)
# from finance_agent/.env before importing the agent, since the ADK/genai
# client reads these at import/instantiation time.
load_dotenv(Path(__file__).parent / "finance_agent" / ".env")

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


from google import genai
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types

from finance_agent.agent import root_agent

logger = logging.getLogger("finance_agent.server")
logging.basicConfig(level=logging.INFO)

APP_NAME = "finance_agent"
DEFAULT_TITLE = "New Conversation"
TITLE_MODEL = "gemini-3.6-flash"

# ---------------------------------------------------------------------------
# ADK runner + in-memory session service (single process, dev-friendly)
# ---------------------------------------------------------------------------
session_service = InMemorySessionService()
runner = Runner(
    app_name=APP_NAME,
    agent=root_agent,
    session_service=session_service,
    # Auto-create the ADK session on first message for a given
    # (user_id, session_id) pair instead of requiring an explicit create step.
    auto_create_session=True,
)

RUN_CONFIG = RunConfig(streaming_mode=StreamingMode.SSE)

# A lightweight genai client used only for generating short conversation
# titles. Reuses the same Vertex AI / API-key configuration loaded above.
title_client = genai.Client()

# Tracks in-flight title-generation tasks keyed by session_id, so the polling
# endpoint (`GET /sessions/{session_id}/title`) can report "pending" vs.
# "ready" without re-triggering generation.
_title_tasks: dict[str, asyncio.Task] = {}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Finance Agent API", version="1.0.0")

# Comma-separated list of allowed origins (e.g. the Finance_UI BFF's origin).
# Defaults to the local Next.js dev server. Matters less once this service is
# private + secret-gated (browsers can't reach it directly), but it's a
# cheap defense-in-depth layer.
_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGIN", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth dependencies
#
# This agent is intended to be called *only* by the Finance_UI Next.js
# backend (a trusted BFF/server-side proxy), never directly by a browser.
# Finance_UI's server-side proxy authenticates end users itself (Google OAuth
# + a Firestore allowlist) and then forwards two headers on every request:
#
#   - X-Internal-Secret : a shared secret proving the caller is Finance_UI's
#                          backend (not a random client that guessed our URL).
#   - X-User-Email       : the *verified* identity of the end user, which we
#                          use as the effective user_id — never anything a
#                          browser client could forge in a request body.
# ---------------------------------------------------------------------------
AGENT_INTERNAL_SECRET = os.environ.get("AGENT_INTERNAL_SECRET")


async def verify_internal_secret(
    x_internal_secret: Optional[str] = Header(default=None, alias="X-Internal-Secret"),
) -> None:
    """Ensures the caller is the trusted Finance_UI backend.

    Fails closed: if AGENT_INTERNAL_SECRET isn't configured on this service at
    all, every protected route rejects with 401 rather than silently allowing
    unauthenticated traffic through.
    """
    if not AGENT_INTERNAL_SECRET or x_internal_secret != AGENT_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


async def get_effective_user_id(
    x_user_email: Optional[str] = Header(default=None, alias="X-User-Email"),
) -> Optional[str]:
    """Returns the verified end-user identity forwarded by Finance_UI.

    This takes precedence over any `user_id` field in the request body/query
    string, which is kept only for backward-compatible local curl testing.
    """
    return x_user_email



class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message/query for the agent.")
    user_id: str = Field(default="default-user", description="Stable user identifier.")
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Conversation/session identifier. If omitted, a new session id is "
            "generated and returned in the stream's first event."
        ),
    )


class CreateSessionRequest(BaseModel):
    user_id: str = Field(default="default-user", description="Stable user identifier.")
    session_id: Optional[str] = Field(
        default=None, description="Optional client-provided session id."
    )
    title: Optional[str] = Field(
        default=None, description="Optional initial title. Defaults to 'New Conversation'."
    )


class SessionSummary(BaseModel):
    session_id: str
    title: str
    last_update_time: float


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class ChatMessage(BaseModel):
    role: str
    text: str
    timestamp: float


class SessionDetailResponse(BaseModel):
    session_id: str
    title: str
    last_update_time: float
    messages: list[ChatMessage]


class TitleStatusResponse(BaseModel):
    status: str  # "pending" | "ready" | "unknown"
    title: Optional[str] = None


# ---------------------------------------------------------------------------
# Title generation helpers
# ---------------------------------------------------------------------------
def _extract_title(session: Session) -> str:
    """Returns the persisted title for a session, or a sensible fallback."""
    title = session.state.get("title")
    if title:
        return title

    # Fall back to a truncated version of the first user message so the
    # sidebar never shows a blank/placeholder title once real content exists.
    for event in session.events:
        if event.author == "user" and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    text = part.text.strip()
                    return (text[:57] + "...") if len(text) > 60 else text
    return DEFAULT_TITLE


async def generate_session_title(message: str) -> str:
    """Calls the model to produce a short, descriptive conversation title."""
    prompt = (
        "Generate a very short, descriptive title (max 6 words, no quotes, "
        "no trailing punctuation) for a chat conversation that starts with "
        f"this user message:\n\n{message}\n\nRespond with ONLY the title text."
    )
    try:
        response = await title_client.aio.models.generate_content(
            model=TITLE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=200,
                temperature=0.3,
                # Disable "thinking" for this call: it's a trivial task and
                # thinking tokens would otherwise eat into max_output_tokens,
                # sometimes leaving no budget left for the actual title text
                # (observed as an empty response with finish_reason=MAX_TOKENS).
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        title = (response.text or "").strip().strip('"').strip("'")
        return title[:80] if title else DEFAULT_TITLE

    except Exception:  # noqa: BLE001
        logger.exception("Failed to generate session title")
        return DEFAULT_TITLE


async def _persist_session_title(
    app_name: str, user_id: str, session_id: str, title: str
) -> None:
    """Writes the generated title into the session's persisted state."""
    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if session is None:
        return
    event = Event(
        invocation_id=str(uuid.uuid4()),
        author="system",
        actions=EventActions(state_delta={"title": title}),
    )
    await session_service.append_event(session, event)


def _schedule_title_generation(
    user_id: str, session_id: str, message: str
) -> asyncio.Task:
    """Kicks off background title generation + persistence for a new session."""

    async def _run() -> str:
        title = await generate_session_title(message)
        await _persist_session_title(APP_NAME, user_id, session_id, title)
        return title

    task = asyncio.create_task(_run())
    _title_tasks[session_id] = task
    return task


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------
def _sse(payload: dict[str, Any]) -> str:
    """Formats a dict payload as a single Server-Sent-Event message."""
    return f"data: {json.dumps(payload, default=str)}\n\n"


def _event_to_chunks(event: Event) -> list[dict[str, Any]]:
    """Normalizes a single ADK Event into zero or more SSE-ready chunks.

    An Event can carry multiple content parts (thoughts, tool calls, tool
    results, text) so we split it into one chunk per part to make the stream
    easy for a client UI to render distinctly.
    """
    chunks: list[dict[str, Any]] = []
    author = event.author or "agent"

    if not event.content or not event.content.parts:
        return chunks

    for part in event.content.parts:
        # --- Thinking / reasoning trace ---
        if getattr(part, "thought", False):
            if part.text:
                chunks.append({
                    "type": "thought",
                    "author": author,
                    "text": part.text,
                    "partial": bool(event.partial),
                })
            continue

        # --- Tool invocation request ---
        if part.function_call:
            chunks.append({
                "type": "tool_call",
                "author": author,
                "name": part.function_call.name,
                "args": part.function_call.args,
                "call_id": part.function_call.id,
            })
            continue

        # --- Tool execution result ---
        if part.function_response:
            chunks.append({
                "type": "tool_result",
                "author": author,
                "name": part.function_response.name,
                "result": part.function_response.response,
                "call_id": part.function_response.id,
            })
            continue

        # --- Plain model text (partial streaming chunk or aggregated) ---
        if part.text:
            chunks.append({
                "type": "text",
                "author": author,
                "text": part.text,
                "partial": bool(event.partial),
            })
            continue

    return chunks


async def run_agent_stream(
    user_id: str, session_id: str, message: str
) -> AsyncGenerator[str, None]:
    """Runs the agent and yields SSE-formatted strings for each event chunk."""
    yield _sse({"type": "session", "session_id": session_id})

    # Detect whether this is the very first message of a new conversation so
    # we know whether to kick off background title generation.
    existing_session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    is_new_session = existing_session is None or len(existing_session.events) == 0

    title_task: Optional[asyncio.Task] = None
    if is_new_session:
        title_task = _schedule_title_generation(user_id, session_id, message)

    new_message = types.Content(role="user", parts=[types.Part(text=message)])

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message,
            run_config=RUN_CONFIG,
        ):
            for chunk in _event_to_chunks(event):
                yield _sse(chunk)

            # Emit a clearly-labeled "final" chunk for the turn's complete
            # user-facing response, in addition to the streamed text parts.
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(
                    p.text or "" for p in event.content.parts if p.text and not getattr(p, "thought", False)
                )
                if final_text:
                    yield _sse({
                        "type": "final",
                        "author": event.author or "agent",
                        "text": final_text,
                    })

        # If we kicked off title generation, give it a short window to finish
        # so the UI can rename the sidebar entry immediately. If it takes
        # longer than that, the client can fall back to polling
        # `GET /sessions/{session_id}/title`.
        if title_task is not None:
            try:
                title = await asyncio.wait_for(title_task, timeout=10)
                yield _sse({
                    "type": "session_title",
                    "session_id": session_id,
                    "title": title,
                })
            except asyncio.TimeoutError:
                logger.info(
                    "Title generation for session %s is taking longer than "
                    "expected; client should poll /sessions/%s/title.",
                    session_id,
                    session_id,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Title generation task failed for session %s", session_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error while running agent")
        yield _sse({"type": "error", "message": str(exc)})
    finally:
        _title_tasks.pop(session_id, None)
        yield _sse({"type": "done"})


@app.post("/chat/stream", dependencies=[Depends(verify_internal_secret)])
async def chat_stream(
    request: ChatRequest,
    effective_user_id: Optional[str] = Depends(get_effective_user_id),
) -> StreamingResponse:
    """Streams the agent's thinking, tool usage, and response text as SSE."""
    session_id = request.session_id or str(uuid.uuid4())
    user_id = effective_user_id or request.user_id

    return StreamingResponse(
        run_agent_stream(
            user_id=user_id,
            session_id=session_id,
            message=request.message,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



# ---------------------------------------------------------------------------
# Session management routes
# ---------------------------------------------------------------------------
@app.post(
    "/sessions",
    response_model=SessionSummary,
    dependencies=[Depends(verify_internal_secret)],
)
async def create_session(
    request: CreateSessionRequest,
    effective_user_id: Optional[str] = Depends(get_effective_user_id),
) -> SessionSummary:
    """Creates a new, empty chat session (e.g. for a "New Chat" button)."""
    user_id = effective_user_id or request.user_id
    initial_state = {"title": request.title} if request.title else None
    try:
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=request.session_id,
            state=initial_state,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=409, detail=str(exc)) from exc


    return SessionSummary(
        session_id=session.id,
        title=_extract_title(session),
        last_update_time=session.last_update_time,
    )


@app.get(
    "/sessions",
    response_model=SessionListResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def list_sessions(
    user_id: Optional[str] = Query(
        default=None, description="The user whose sessions to list."
    ),
    effective_user_id: Optional[str] = Depends(get_effective_user_id),
) -> SessionListResponse:
    """Lists all sessions for a user, newest first — for a sidebar view."""
    resolved_user_id = effective_user_id or user_id
    if not resolved_user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")
    response = await session_service.list_sessions(
        app_name=APP_NAME, user_id=resolved_user_id
    )


    summaries = [
        SessionSummary(
            session_id=session.id,
            title=_extract_title(session),
            last_update_time=session.last_update_time,
        )
        for session in response.sessions
    ]
    summaries.sort(key=lambda s: s.last_update_time, reverse=True)
    return SessionListResponse(sessions=summaries)


@app.get(
    "/sessions/{session_id}",
    response_model=SessionDetailResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def get_session_detail(
    session_id: str,
    user_id: Optional[str] = Query(default=None, description="The owner of the session."),
    effective_user_id: Optional[str] = Depends(get_effective_user_id),
) -> SessionDetailResponse:
    """Returns the full message history for a single session."""
    resolved_user_id = effective_user_id or user_id
    if not resolved_user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=resolved_user_id, session_id=session_id
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")


    messages: list[ChatMessage] = []
    for event in session.events:
        if not event.content or not event.content.parts:
            continue
        text = "".join(
            p.text or "" for p in event.content.parts
            if p.text and not getattr(p, "thought", False)
        )
        if not text:
            continue
        role = "user" if event.author == "user" else "assistant"
        messages.append(ChatMessage(role=role, text=text, timestamp=event.timestamp))

    return SessionDetailResponse(
        session_id=session.id,
        title=_extract_title(session),
        last_update_time=session.last_update_time,
        messages=messages,
    )


@app.delete("/sessions/{session_id}", dependencies=[Depends(verify_internal_secret)])
async def delete_session(
    session_id: str,
    user_id: Optional[str] = Query(default=None, description="The owner of the session."),
    effective_user_id: Optional[str] = Depends(get_effective_user_id),
) -> dict[str, str]:
    """Deletes a session and all of its history."""
    resolved_user_id = effective_user_id or user_id
    if not resolved_user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")
    await session_service.delete_session(
        app_name=APP_NAME, user_id=resolved_user_id, session_id=session_id
    )
    _title_tasks.pop(session_id, None)
    return {"status": "deleted", "session_id": session_id}



@app.get(
    "/sessions/{session_id}/title",
    response_model=TitleStatusResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def get_session_title(
    session_id: str,
    user_id: Optional[str] = Query(default=None, description="The owner of the session."),
    effective_user_id: Optional[str] = Depends(get_effective_user_id),
) -> TitleStatusResponse:
    """Polls for the async-generated title of a session.

    Useful for a UI that already rendered a placeholder sidebar entry (e.g.
    right after sending the first message) and wants to refresh it once the
    real title is ready, without keeping the SSE connection open.
    """
    resolved_user_id = effective_user_id or user_id
    if not resolved_user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")

    task = _title_tasks.get(session_id)
    if task is not None:
        if task.done():
            try:
                return TitleStatusResponse(status="ready", title=task.result())
            except Exception:  # noqa: BLE001
                logger.exception("Title generation task raised for session %s", session_id)
                return TitleStatusResponse(status="ready", title=DEFAULT_TITLE)
        return TitleStatusResponse(status="pending")

    session = await session_service.get_session(
        app_name=APP_NAME, user_id=resolved_user_id, session_id=session_id
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")


    title = session.state.get("title")
    if title:
        return TitleStatusResponse(status="ready", title=title)
    return TitleStatusResponse(status="unknown", title=_extract_title(session))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def main():
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
