# Finance Agent

An AI-powered **Equity Research Analyst** agent built with [Google's Agent Development Kit (ADK)](https://github.com/google/adk-python), backed by live market data from `yfinance`, and exposed over HTTP via a **FastAPI streaming API**. The API streams the agent's internal "thinking", tool calls, tool results, and generated text in real time using Server-Sent Events (SSE) — making it straightforward to plug into any custom chatbot UI.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How the Agent Works](#how-the-agent-works)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Install dependencies](#1-install-dependencies)
  - [2. Configure Google Cloud / Vertex AI credentials](#2-configure-google-cloud--vertex-ai-credentials)
  - [3. Environment variables](#3-environment-variables)
- [Running the Server](#running-the-server)
- [Authentication & Trust Model](#authentication--trust-model)
  - [Shared secret (`X-Internal-Secret`)](#shared-secret-x-internal-secret)
  - [Verified user identity (`X-User-Email`)](#verified-user-identity-x-user-email)
  - [CORS](#cors)
  - [Cloud Run IAM (deployment-only)](#cloud-run-iam-deployment-only)
- [API Reference](#api-reference)
  - [`POST /chat/stream`](#post-chatstream)
  - [Session Management Routes](#session-management-routes)
  - [`GET /health`](#get-health)
  - [SSE Event Types](#sse-event-types)

- [Connecting a Custom Chat UI](#connecting-a-custom-chat-ui)
  - [Why POST + SSE instead of `EventSource`](#why-post--sse-instead-of-eventsource)
  - [Minimal JavaScript client example](#minimal-javascript-client-example)
  - [React hook example](#react-hook-example)
  - [Rendering guidance per event type](#rendering-guidance-per-event-type)
  - [Session/conversation management](#sessionconversation-management)
- [Testing with curl](#testing-with-curl)
- [Extending the Agent](#extending-the-agent)
- [Notes & Limitations](#notes--limitations)

---

## Overview

This project wraps a single ADK `Agent` (`finance_agent/agent.py`) configured as an **equity research analyst**. Given a stock ticker or company name, it:

1. Gathers data using purpose-built tools (`yfinance_tools/finance_tools.py`):
   - `get_company_info` — sector, market cap, valuation multiples, business summary
   - `get_historical_market_data` — recent price action (capped at 3 months by the agent's instructions to control context size)
   - `get_fundamental_data` — income statement / balance sheet / cash flow (most recent fiscal years)
   - `get_corporate_actions` — dividend and stock split history
2. Cross-references valuation, fundamentals, and price momentum.
3. Produces a structured markdown investment report (Executive Summary, Valuation Overview, Fundamental Health, Price Action, Catalysts & Risks, Disclaimer).

The agent is served through `main.py`, a FastAPI app that streams every step of the agent's reasoning process — not just the final answer — so a frontend can show a live "the agent is thinking / calling a tool / writing the answer" experience.

---

## Project Structure

```
Finance_Agent/
├── main.py                       # FastAPI app: SSE streaming endpoint wrapping the ADK Runner
├── finance_agent/
│   ├── agent.py                  # ADK Agent definition (model, instructions, tools, thinking config)
│   ├── .env                      # Vertex AI / Google Cloud config + auth secrets (not committed)
│   └── __init__.py
├── yfinance_tools/
│   └── finance_tools.py          # Tool functions the agent can call (wraps yfinance)
├── pyproject.toml                # Project dependencies (managed with uv)
└── README.md
```

---

## How the Agent Works

The agent is instantiated in `finance_agent/agent.py` using ADK's `Agent` (an `LlmAgent`):

```python
root_agent = Agent(
    model='gemini-3.6-flash',
    name='equity_research_agent',
    description='An expert equity research analyst ...',
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True),
    ),
    instruction='''...detailed system prompt...''',
    tools=[
        get_company_info,
        get_historical_market_data,
        get_fundamental_data,
        get_corporate_actions,
    ],
)
```

Key details:

- **`thinking_config=ThinkingConfig(include_thoughts=True)`** tells the Gemini model to emit its internal reasoning as distinct "thought" parts in the response stream. Without this, only the final text and tool calls would be visible — no reasoning trace.
- **Tools** are plain Python functions with docstrings; ADK auto-generates the function-calling schema from their signatures and docstrings.
- The **system instruction** enforces strict context-window discipline (e.g. never pull more than 3 months of historical data, only the last 2 fiscal years of fundamentals) and a fixed markdown output format.

`main.py` drives this agent using ADK's `Runner`:

```python
runner = Runner(
    app_name="finance_agent",
    agent=root_agent,
    session_service=InMemorySessionService(),
    auto_create_session=True,
)

async for event in runner.run_async(
    user_id=..., session_id=..., new_message=..., 
    run_config=RunConfig(streaming_mode=StreamingMode.SSE),
):
    ...
```

Each `Event` yielded by `run_async` may contain one or more `Part`s:

| Part attribute        | Meaning                                                   | Streamed as   |
|------------------------|------------------------------------------------------------|---------------|
| `part.thought=True` + `part.text` | Model's internal reasoning                        | `"thought"`   |
| `part.function_call`   | A request to invoke a tool (name + args)                  | `"tool_call"` |
| `part.function_response` | The result returned by a tool                          | `"tool_result"` |
| `part.text` (no `thought` flag) | Regular output text (partial or final)             | `"text"` / `"final"` |

`main.py` normalizes these into simple, UI-friendly JSON chunks and streams them as Server-Sent Events.

---

## Prerequisites

- Python **3.13+**
- [`uv`](https://docs.astral.sh/uv/) package manager
- A Google Cloud project with **Vertex AI API** enabled (this project is configured for Vertex AI, not a raw Gemini API key — see `finance_agent/.env`)
- `gcloud` CLI installed and authenticated

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

This installs everything declared in `pyproject.toml`, including `google-adk`, `yfinance`, `fastapi`, `uvicorn`, and `python-dotenv`.

### 2. Configure Google Cloud / Vertex AI credentials

This project uses **Vertex AI** via Application Default Credentials (ADC), not an API key. Authenticate once per machine:

```bash
gcloud auth application-default login
```

If you see a warning about a missing quota project, set one explicitly (use your own GCP project id):

```bash
gcloud auth application-default set-quota-project <your-gcp-project-id>
```

### 3. Environment variables

`finance_agent/.env` configures the ADK/genai client to use Vertex AI, plus the auth settings for BFF integration:

```env
GOOGLE_GENAI_USE_ENTERPRISE=1
GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
GOOGLE_CLOUD_LOCATION=global

# Shared secret — must match Finance_UI's AGENT_INTERNAL_SECRET.
# Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"
AGENT_INTERNAL_SECRET=change-me-to-a-real-secret

# Comma-separated allowed CORS origins (defense-in-depth).
ALLOWED_ORIGIN=http://localhost:3000
```

`main.py` automatically loads this file on startup via `python-dotenv`, before the agent module is imported, so these variables are available to the Vertex AI client.

> **Important:** `AGENT_INTERNAL_SECRET` must be set to a real value before any protected route will accept requests (the server fails closed — see [Authentication & Trust Model](#authentication--trust-model) below).

---

## Running the Server

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or simply:

```bash
uv run main.py
```

(`main.py`'s `if __name__ == "__main__"` block also boots uvicorn on `0.0.0.0:8000`.)

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## Authentication & Trust Model

This agent is designed to be called **only by the Finance_UI Next.js backend** (a trusted BFF / server-side proxy), never directly by a browser. Finance_UI authenticates end users itself (Google OAuth + a Firestore allowlist) and then forwards two trusted headers on every proxied request.

### Shared secret (`X-Internal-Secret`)

Every protected route (all routes except `GET /health`) requires an `X-Internal-Secret` header whose value matches the `AGENT_INTERNAL_SECRET` environment variable configured on both the agent and the UI backend.

- **Fail-closed:** if `AGENT_INTERNAL_SECRET` is unset or the header is missing/mismatched, the route returns `401 Unauthorized`.
- In local development, both services read the same secret from their respective `.env` files. In Cloud Run, both get it from the same Secret Manager secret.

### Verified user identity (`X-User-Email`)

Finance_UI also sends `X-User-Email: <verified-email>` on every request — the authenticated, allowlist-checked email address of the end user. The agent uses this as the **effective `user_id`** for all session operations, rather than trusting the JSON body's `user_id` field.

- When `X-User-Email` is present, it takes precedence over `user_id` in the request body or query string.
- The body/query `user_id` fields are kept for backward compatibility and local curl testing (when the header is absent, the body value is used as a fallback).
- This is the mechanism that prevents user A from reading/deleting user B's sessions — each user's sessions are keyed by their verified email.

### CORS

`allow_origins` is restricted to the value(s) in the `ALLOWED_ORIGIN` environment variable (comma-separated, defaults to `http://localhost:3000`). In practice this matters less once the service is private + secret-gated (browsers can't reach it directly), but it's a cheap defense-in-depth layer.

### Cloud Run IAM (deployment-only)

For production Cloud Run deployment, an additional layer is recommended (no code changes required):

- Deploy `finance-agent` with `--no-allow-unauthenticated`.
- Grant the Finance_UI Cloud Run service's service account `roles/run.invoker` on the agent service.
- Cloud Run's platform layer then enforces a valid Google-signed ID token automatically, on top of the shared secret.
- Finance_UI's `agentProxy.ts` already attaches this token when it detects a `*.run.app` URL.

---

## API Reference

> **Note:** All routes except `GET /health` require the `X-Internal-Secret` header. The `X-User-Email` header should be provided to identify the end user; if absent, the `user_id` from the body/query is used as a fallback.

### `POST /chat/stream`

Streams the agent's full reasoning process for a single user message as Server-Sent Events.

**Required headers:**

| Header             | Description                                      |
|--------------------|--------------------------------------------------|
| `X-Internal-Secret` | Must match the server's `AGENT_INTERNAL_SECRET`. |
| `X-User-Email`      | Verified email of the end user (used as `user_id`). |

**Request body (JSON):**

| Field        | Type   | Required | Description                                                                 |
|--------------|--------|----------|-------------------------------------------------------------------------------|
| `message`    | string | yes      | The user's message/query.                                                    |
| `user_id`    | string | no       | Fallback identifier (used only if `X-User-Email` header is absent). Defaults to `"default-user"`. |
| `session_id` | string | no       | Conversation id. Omit to start a new conversation; a new id is generated and returned as the first stream event. Pass the same id back on subsequent turns to continue the conversation with memory. |

**Response:** `Content-Type: text/event-stream`, an SSE stream where each message is:

```
data: {"type": "...", ...}\n\n
```

### Session Management Routes

These routes back a chat sidebar UI: creating new conversations, listing past ones with titles, loading full history, and deleting conversations. All session state lives in the same in-memory `InMemorySessionService` used by `/chat/stream`, keyed by `(app_name, user_id, session_id)`.

> All session routes require `X-Internal-Secret` and use `X-User-Email` as the effective `user_id`. The `user_id` query parameter / body field serves as a fallback when the header is absent.

#### `POST /sessions`

Creates a new, empty session (e.g. for an explicit "New Chat" button, as opposed to lazily auto-creating one via the first `/chat/stream` call).

**Request body (JSON):**

| Field        | Type   | Required | Description |
|--------------|--------|----------|--------------|
| `user_id`    | string | no       | Fallback (used only if `X-User-Email` absent). Defaults to `"default-user"`. |
| `session_id` | string | no       | Optional client-provided id. Auto-generated if omitted. |
| `title`      | string | no       | Optional initial title. Defaults to `"New Conversation"` until auto-titled. |

**Response:** `SessionSummary` — `{ "session_id", "title", "last_update_time" }`.

#### `GET /sessions?user_id=...`

Lists all session summaries for a user, sorted newest-first by `last_update_time` — this is what you render in the sidebar. The `user_id` query param is a fallback; `X-User-Email` takes precedence.

**Response:**
```json
{ "sessions": [ { "session_id": "...", "title": "...", "last_update_time": 1234567890.1 }, ... ] }
```

#### `GET /sessions/{session_id}?user_id=...`

Returns the full chat history for one session (for loading a conversation when the user clicks it in the sidebar).

**Response:**
```json
{
  "session_id": "...",
  "title": "...",
  "last_update_time": 1234567890.1,
  "messages": [
    { "role": "user", "text": "...", "timestamp": 1234567890.1 },
    { "role": "assistant", "text": "...", "timestamp": 1234567891.2 }
  ]
}
```

Returns `404` if the session doesn't exist for that user.

#### `DELETE /sessions/{session_id}?user_id=...`

Deletes a session and its full history. Returns `{"status": "deleted", "session_id": "..."}`.

#### `GET /sessions/{session_id}/title?user_id=...`

Polls the status of the session's auto-generated title. Useful if your UI doesn't want to wait on the `/chat/stream` connection for the `session_title` event (see below) — e.g. if you rendered a placeholder sidebar row immediately after sending the first message and want to refresh it later.

**Response:**
```json
{ "status": "pending" | "ready" | "unknown", "title": "..." }
```
- `"pending"` — generation is still running in the background.
- `"ready"` — `title` contains the generated (and persisted) title.
- `"unknown"` — no title-generation task is tracked for this session (e.g. after a server restart); `title` falls back to the first user message or `"New Conversation"`.

> **How titles are generated:** The first time a brand-new session receives a message via `/chat/stream`, the server kicks off a background call to the model asking for a short (≤6 word) title based on that first message, and persists it into the session's state once ready. The streaming response waits up to ~10 seconds for this to finish and emits it as a `session_title` event; if it's slower than that, poll `GET /sessions/{session_id}/title` instead.

### `GET /health`

Simple liveness check, returns `{"status": "ok"}`. **No auth required** — this endpoint is open so Cloud Run health checks work without credentials.

### SSE Event Types

| `type`          | Fields                                                | Description |
|-----------------|--------------------------------------------------------|--------------|
| `session`       | `session_id`                                           | First event of every stream; carries the session id to reuse on the next turn. |
| `thought`       | `author`, `text`, `partial`                            | A chunk of the model's internal reasoning. |
| `tool_call`     | `author`, `name`, `args`, `call_id`                    | The agent is invoking a tool. |
| `tool_result`   | `author`, `name`, `result`, `call_id`                  | The result returned by a tool call. |
| `text`          | `author`, `text`, `partial`                            | A chunk of the model's user-facing answer (streamed token-by-token when `partial=true`). |
| `final`         | `author`, `text`                                       | The complete, assembled final answer for the turn. |
| `session_title` | `session_id`, `title`                                  | Emitted once, only for a brand-new session, after the auto-generated title is ready. Use it to rename the sidebar entry in real time. |
| `error`         | `message`                                              | An error occurred while running the agent. |
| `done`          | —                                                       | Sentinel marking the end of the stream. Always sent last. |


> **Note on duplicate text:** With SSE streaming mode, you will receive both partial `text` chunks (for a typewriter effect) **and** one final aggregated `text` event, followed by a `final` event with the same content. A UI should either (a) render only `partial: true` text chunks incrementally and ignore the aggregated duplicate, or (b) ignore all `text` events and just render the single `final` event once it arrives. See [Rendering guidance](#rendering-guidance-per-event-type) below.

---

## Connecting a Custom Chat UI

### Why POST + SSE instead of `EventSource`

The native browser `EventSource` API only supports `GET` requests and cannot send a JSON body, so it can't be used directly against `POST /chat/stream`. Instead, use `fetch()` with a `ReadableStream` reader, which works with any HTTP method and is well supported in modern browsers and Node.

> **Important:** In the recommended architecture, the browser never calls this agent directly. Instead, your Next.js (or similar) backend acts as a BFF proxy — it authenticates the user, then forwards the request to this agent with the `X-Internal-Secret` and `X-User-Email` headers. The examples below show the raw fetch pattern for illustration; in production the `user_id` body field is ignored in favor of `X-User-Email`.

### Minimal JavaScript client example

```javascript
async function streamChat(message, sessionId, onEvent) {
  const response = await fetch("http://localhost:8000/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Internal-Secret": "<your-shared-secret>",
      "X-User-Email": "user@example.com",
    },
    body: JSON.stringify({
      message,
      session_id: sessionId, // pass null/undefined for a new conversation
    }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE messages are separated by a blank line ("\n\n")
    const parts = buffer.split("\n\n");
    buffer = parts.pop(); // last part may be incomplete, keep it in the buffer

    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      const jsonStr = line.slice("data:".length).trim();
      const event = JSON.parse(jsonStr);
      onEvent(event);
    }
  }
}

// Usage:
let currentSessionId = null;

streamChat("Give me a quick analysis of AAPL", currentSessionId, (event) => {
  switch (event.type) {
    case "session":
      currentSessionId = event.session_id; // save for the next turn
      break;
    case "thought":
      console.log("🤔", event.text);
      break;
    case "tool_call":
      console.log("🔧 calling", event.name, event.args);
      break;
    case "tool_result":
      console.log("✅ result from", event.name, event.result);
      break;
    case "text":
      if (event.partial) process.stdout.write(event.text); // typewriter effect
      break;
    case "final":
      console.log("\n\n=== FINAL ===\n", event.text);
      break;
    case "error":
      console.error("Error:", event.message);
      break;
    case "done":
      console.log("Stream finished.");
      break;
  }
});
```

### React hook example

```jsx
import { useCallback, useRef, useState } from "react";

export function useFinanceAgentChat() {
  const [messages, setMessages] = useState([]); // { role, content, thoughts, toolCalls }
  const sessionIdRef = useRef(null);

  const sendMessage = useCallback(async (text) => {
    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", thoughts: [], toolCalls: [] },
    ]);

    const response = await fetch("http://localhost:8000/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Internal-Secret": "<your-shared-secret>",
        "X-User-Email": "user@example.com",
      },
      body: JSON.stringify({
        message: text,
        session_id: sessionIdRef.current,
      }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const updateLastAssistant = (updater) => {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        next[next.length - 1] = updater(last);
        return next;
      });
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const part of parts) {
        if (!part.startsWith("data:")) continue;
        const event = JSON.parse(part.slice(5).trim());

        if (event.type === "session") {
          sessionIdRef.current = event.session_id;
        } else if (event.type === "thought") {
          updateLastAssistant((m) => ({ ...m, thoughts: [...m.thoughts, event.text] }));
        } else if (event.type === "tool_call") {
          updateLastAssistant((m) => ({
            ...m,
            toolCalls: [...m.toolCalls, { name: event.name, args: event.args }],
          }));
        } else if (event.type === "final") {
          updateLastAssistant((m) => ({ ...m, content: event.text }));
        } else if (event.type === "error") {
          updateLastAssistant((m) => ({ ...m, content: `⚠️ ${event.message}` }));
        }
      }
    }
  }, []);

  return { messages, sendMessage };
}
```

### Rendering guidance per event type

A typical chat UI layout for one assistant turn:

1. **Collapsible "Thinking..." panel** — append every `thought` chunk here as it streams in; collapse/hide once `final` arrives.
2. **Tool activity log** — for each `tool_call`, show a small "Calling `get_company_info(ticker_symbol=AAPL)`..." indicator; when the matching `tool_result` (same `call_id`) arrives, mark it as complete (optionally show a summarized result).
3. **Main answer bubble** — either:
   - Append `text` chunks where `partial: true` for a live typewriter effect (recommended for best UX), **or**
   - Wait and render only the single `final` event (simplest, no risk of duplicate text).
4. **`done`** — stop any "typing..." indicator/spinner.
5. **`error`** — show an inline error message in the chat bubble.

### Session/conversation management

- On the **first message** of a new conversation, omit `session_id` (or send `null`).
- The server responds with a `session` event containing a generated `session_id` — store it (e.g., in component state, `localStorage`, or a cookie).
- Send that same `session_id` on every subsequent message in the conversation so the agent retains full context/history.
- Sessions are stored **in-memory only** (`InMemorySessionService`) — they are lost when the server restarts. For persistence across restarts, swap in ADK's `DatabaseSessionService` or `VertexAiSessionService`.

---

## Testing with curl

All protected routes require the `X-Internal-Secret` header. For local development, use the value from your `finance_agent/.env`:

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-Internal-Secret: change-me-to-a-real-secret" \
  -H "X-User-Email: demo@example.com" \
  -d '{"message": "Give me a quick analysis of AAPL"}'
```

The `-N` flag disables curl's output buffering so you see events as they stream in, rather than all at once at the end.

To continue the same conversation, copy the `session_id` from the first `session` event and include it in the next request:

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-Internal-Secret: change-me-to-a-real-secret" \
  -H "X-User-Email: demo@example.com" \
  -d '{"message": "What about its main competitor, MSFT?", "session_id": "<paste-session-id-here>"}'
```

> **Note:** The `user_id` body field is still accepted as a fallback when `X-User-Email` is absent, but in the BFF architecture the header always takes precedence.

The health endpoint requires no authentication:

```bash
curl http://127.0.0.1:8000/health
```

---

## Extending the Agent

- **Add a new tool:** write a plain Python function with type hints and a clear docstring in `yfinance_tools/finance_tools.py`, then add it to the `tools=[...]` list in `finance_agent/agent.py`. ADK automatically builds the function-calling schema from the signature/docstring.
- **Change the model:** update the `model=` field in `finance_agent/agent.py`. Note: thought streaming (`ThinkingConfig`) requires a Gemini model that supports "thinking".
- **Persist sessions:** replace `InMemorySessionService()` in `main.py` with `DatabaseSessionService` (SQLite/Postgres) or `VertexAiSessionService` for durability across restarts and multi-instance deployments.

---

## Notes & Limitations

- This server runs a **single global `Runner`/session service instance** — fine for local development and single-process deployments. For production, consider a persistent session backend and running behind a proper ASGI server (e.g. `uvicorn` with multiple workers behind a load balancer, noting that `InMemorySessionService` is not shared across processes).
- **Cloud Run scaling:** Since sessions are stored in-memory, the Cloud Run service should be deployed with `min-instances=1` and `max-instances=1` to prevent session loss from instance replacement or scale-up. For multi-instance deployments, switch to a persistent session backend.
- CORS is restricted to the `ALLOWED_ORIGIN` environment variable (defaults to `http://localhost:3000`). In production, the service should be private + secret-gated so browsers can't reach it directly anyway.
- All protected routes require the `X-Internal-Secret` header. The server **fails closed**: if the `AGENT_INTERNAL_SECRET` environment variable is not set, every protected route returns `401 Unauthorized`.
- The `user_id` in request bodies/query params is a fallback only. When the `X-User-Email` header is present (as it always is in the BFF architecture), it takes precedence as the effective user identity, preventing cross-user session access.
- The agent's system instructions cap historical data requests to 3 months and fundamentals to the most recent 2 fiscal years to avoid exhausting the model's context window; adjust these constraints in `finance_agent/agent.py` if you need deeper historical analysis.
