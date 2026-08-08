# multi-user-doc-rag

Multi-user document search & conversational Q&A over a handful of companies'
earnings-call transcripts, with per-user access control: each user is scoped
to the company (or companies) their dummy account is authorized for, enforced
at the vector-search layer itself.

## Table of contents

- [Features](#features)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Architecture](#architecture)
- [Getting started](#getting-started)
- [Using the API](#using-the-api)
- [Running with Docker](#running-with-docker)
- [Testing](#testing)
- [Access control model](#access-control-model)
- [Import paths gotcha](#import-paths)
- [Further reading](#further-reading)

## Features

- Retrieval scoped per user at the vector-search layer, not as a post-filter
  or a UI-level restriction.
- Multi-turn conversational Q&A with per-conversation history, isolated per
  user so concurrent sessions never collide or leak.
- Multi-company questions are decomposed and answered per company in
  parallel, then stitched into one reply.
- Streaming (SSE) answers alongside a non-streaming JSON endpoint.
- Versioned prompts, so a prompt can be rolled forward/back without losing
  the previous copy.
- Optional LangSmith tracing of every graph node and LLM call, and file +
  console logging when tracing isn't configured.

## Tech stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI, Pydantic v2, `pydantic-settings` |
| Conversational pipeline | LangGraph |
| LLM client | OpenAI-compatible client (`openai` SDK) against Groq or OpenRouter |
| Embeddings | OpenRouter embeddings API |
| Parsing | Docling (PDF -> markdown) |
| Vector store / history store | MongoDB Atlas (`$vectorSearch`) |
| Auth | PyJWT |
| Observability | LangSmith tracing (optional) + rotating file logs |
| Frontend | React 19 + TypeScript, Vite |
| Tests | pytest |

## Project structure

```
backend/src/
  api/          FastAPI app, routes, JWT auth, request/response schemas
  config/       settings, per-user access-control map, logging, LangSmith setup
  graph/        LangGraph conversational pipeline (nodes, routing, state)
  ingest/       PDF -> markdown -> chunks -> embeddings -> MongoDB Atlas
  llm/          OpenAI-compatible chat-completion client
  prompts/      versioned prompt templates + registry
  retrieval/    ACL-filtered vector search
  store/        Mongo client, vector search, conversation history
frontend/src/    React + TypeScript chat UI (login screen, chat screen)
data/            sample PDFs, plus generated parsed/ and chunks/ JSON
docs/            implementation deep-dive and system dossier
tests/           pytest suite with Mongo/LLM fakes
```

## Architecture

- **Parsing/chunking** (`backend/src/ingest/`): PDFs -> markdown (Docling) ->
  speaker-turn chunks, each tagged with `company_id`.
- **Embeddings**: OpenRouter embeddings API (`nvidia/nemotron-3-embed-1b:free`
  by default).
- **Vector store**: MongoDB Atlas, using the `$vectorSearch` aggregation
  stage. Every chunk document carries its `company_id`; the Atlas Search
  index declares `company_id` as a `filter` field so a query only ever
  searches (never just post-filters) the documents the caller is authorized
  for.
- **Auth**: dummy email login (`backend/src/config/users.py` maps email ->
  authorized `company_id`s) issuing a JWT that carries that company list.
  Every retrieval request re-derives its allowed companies from the JWT, not
  from client input.
- **Conversational Q&A** (`backend/src/graph/`): a LangGraph pipeline behind
  `POST /api/conversations/{conv_id}/messages` (and its SSE sibling
  `.../messages/stream`) -- `classify -> rephrase -> fetch -> build_answer ->
  guardrail`. The classifier only routes UX (greeting / out-of-scope / real
  question); it is **not** a security control. `fetch` is the one
  deterministic ACL pre-filter, and it always uses companies re-looked-up
  fresh from `config/users.py` for the caller's email -- never the
  `companies` claim cached inside the JWT -- so a still-valid token issued
  before a permission change can't leak access. Conversation turns are
  stored in MongoDB, keyed by `user_email::conv_id` (see
  `backend/src/store/history_store.py`), so concurrent users' histories can
  never collide or leak into each other.

  <p align="center">
    <img src="arch/langgraph_workflow.png" alt="LangGraph conversational Q&A workflow" width="480">
  </p>

  `classify` routes to either the `continue` branch (`rephrase -> decompose ->
  fetch_one -> answer_one -> combine_answer -> guardrail`) or straight to
  `canned_response` for a greeting/out-of-scope message (`greet`/`deny`).
  `decompose` splits a question naming several companies into one
  sub-question per company; `fetch_one` and then `answer_one` fan out **one
  concurrent run per company** via LangGraph's `Send` API (the boxed lanes
  above -- each company's fetch and answer runs in parallel, not
  sequentially), and `combine_answer` fans back in once every branch has
  finished, stitching the per-company answers into a single reply. See
  `backend/src/graph/graph.py` for the full routing logic.
- **Prompts** (`backend/src/prompts/`): each pipeline prompt is versioned as
  `prompts/<name>/vN.txt`, with the active version per prompt pinned in
  `prompts/registry.py::CURRENT_VERSIONS`. Roll a prompt forward by adding a
  new `vN.txt` and bumping the pointer; old versions stay on disk for
  rollback/audit.
- **Observability** (`backend/src/config/observability.py`,
  `config/logger.py`): every graph node and LLM call is optionally traced to
  LangSmith (set `LANGSMITH_API_KEY` to enable; a complete no-op otherwise),
  and every module logs structured `[step] input=...` / `[step] output=...`
  lines to console + a rotating `logs/app.log`. See
  [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md#13-observability) for
  the full breakdown.
- **Frontend** (`frontend/`): a small React + TypeScript (Vite) app -- an
  email-picker login screen (`screens/Login/`, backed by a dummy
  `DEMO_USERS` list mirroring `config/users.py`) and a chat screen
  (`screens/Main/`) with a conversation sidebar, message list, and
  follow-up input, talking to the API above.

## Getting started

### Prerequisites

- Python 3.12+ and Node 20+
- A MongoDB Atlas cluster (see below) -- plain community MongoDB has no
  `$vectorSearch` support
- An [OpenRouter](https://openrouter.ai/keys) API key (embeddings, and
  optionally chat) and/or a [Groq](https://console.groq.com/keys) API key
  (chat, the default provider)

### 1. MongoDB Atlas

Vector search (`$vectorSearch`) requires a real Atlas cluster (the free M0
tier works) or the `mongodb/mongodb-atlas-local` Docker image -- plain
community MongoDB has no vector search support.

1. Create a cluster in [Atlas](https://cloud.mongodb.com).
2. Atlas UI -> **Connect** -> **Drivers** -> copy the `mongodb+srv://...`
   connection string, filling in your database user's password.
3. Atlas UI -> **Network Access** -> add your current IP (or `0.0.0.0/0` for
   local dev only).

### 2. Environment

```bash
cp .env.example .env
```

Fill in `.env`. The full set of tunables lives in
[`backend/src/config/settings.py`](backend/src/config/settings.py); the ones
that matter to get running:

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Embeddings (ingestion + query time); also chat if `CHAT_PROVIDER=openrouter` |
| `SECRET_KEY` | Yes | JWT signing secret -- change the `dev-secret-key-change-me` default before any real deployment |
| `MONGO_DB_STRING` (or `MONGODB_URI`) | Yes | Atlas connection string, password filled in |
| `CHAT_PROVIDER` | No | `groq` (default) or `openrouter` -- picks which OpenAI-compatible API serves chat completions |
| `GROQ_API_KEY` | Required if `CHAT_PROVIDER=groq` (the default) | Chat completions via Groq |
| `CHAT_MODEL_NAME` | No | Default `llama-3.3-70b-versatile` (a Groq slug); use an OpenRouter chat-capable slug instead if `CHAT_PROVIDER=openrouter` |
| `MONGODB_DB_NAME` / `MONGODB_COLLECTION` / `MONGODB_VECTOR_INDEX` | No | Sensible defaults |
| `MONGODB_HISTORY_COLLECTION` | No | Collection for per-user conversation history |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | No | Optional LangSmith tracing; tracing is a no-op with no key set |

### 3. Install & ingest

Run from the **repo root** (not `backend/` -- the ingest scripts default to
`data/...` paths relative to the repo root):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-ingest.txt   # requirements.txt + docling, for the ingest step only

PYTHONPATH=backend python -m src.ingest.parser        # data/*.pdf -> data/parsed/*.json
PYTHONPATH=backend python -m src.ingest.chunker       # data/parsed/*.json -> data/chunks/*.json
PYTHONPATH=backend python -m src.ingest.embed_and_store   # embeds chunks, upserts into MongoDB Atlas,
                                                            # creates the vector index on first run
```

The vector index takes ~30-60s to become queryable after first creation;
`embed_and_store.py` waits for that automatically.

(`requirements-ingest.txt` is only needed for that one-time ingestion step --
it pulls in Docling and CPU torch/torchvision. The API server itself only
needs `requirements.txt`, which `requirements-ingest.txt` already includes,
so nothing extra to install if you ingested from the same virtualenv.)

### 4. Run the API

```bash
PYTHONPATH=backend uvicorn backend.src.api.main:app --reload
```

### 5. Run the frontend

```bash
cd frontend
cp .env.example .env   # VITE_API_IP / VITE_API_PORT -- must be reachable from the browser
npm install
npm run dev
```

Open the printed Vite URL (default `http://localhost:5173`), pick one of the
demo emails on the login screen, and chat. `frontend/src/screens/Login/demoUsers.ts`
mirrors `backend/src/config/users.py`'s email -> company mapping for display
purposes only -- access control is still enforced entirely server-side.

## Using the API

```bash
# Login as a dummy user (see backend/src/config/users.py for the email -> company map)
curl -s -X POST localhost:8000/api/auth/login -H 'content-type: application/json' \
  -d '{"email": "alice@example.com"}' | tee /tmp/login.json

TOKEN=$(python3 -c "import json;print(json.load(open('/tmp/login.json'))['access_token'])")

# Query -- results are restricted to alice's authorized companies (TCS, Infosys)
curl -s -X POST localhost:8000/api/query -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"query": "What was revenue growth?"}'

# Create a conversation -- returns a conv_id up front (e.g. for a "New Chat" button)
curl -s -X POST localhost:8000/api/conversations -H "authorization: Bearer $TOKEN" | tee /tmp/conv.json

CONV_ID=$(python3 -c "import json;print(json.load(open('/tmp/conv.json'))['conv_id'])")

# Conversational Q&A -- synthesized answer + citations, with follow-up support
curl -s -X POST localhost:8000/api/conversations/$CONV_ID/messages -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"message": "What was revenue growth?"}'

# Same thing, streamed as Server-Sent Events (delta events, then a final `done` event)
curl -s -N -X POST localhost:8000/api/conversations/$CONV_ID/messages/stream -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"message": "What was revenue growth?"}'

# Follow-up in the same conversation -- same conv_id, so history is applied
curl -s -X POST localhost:8000/api/conversations/$CONV_ID/messages -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"message": "and what about margins?"}'

# List this user's conversations
curl -s localhost:8000/api/conversations -H "authorization: Bearer $TOKEN"

# Fetch the full thread for one conversation
curl -s localhost:8000/api/conversations/$CONV_ID -H "authorization: Bearer $TOKEN"
```

Full endpoint list:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/auth/login` | Email -> JWT + authorized companies |
| `GET` | `/api/auth/me` | Current user info from the JWT |
| `POST` | `/api/query` | Retrieval-only (no LLM synthesis); useful for debugging retrieval |
| `GET` | `/api/conversations` | List the caller's conversations |
| `POST` | `/api/conversations` | Create a conversation, returns `conv_id` |
| `GET` | `/api/conversations/{conv_id}` | Full thread for one conversation |
| `POST` | `/api/conversations/{conv_id}/messages` | Send a message, get a synthesized answer + citations |
| `POST` | `/api/conversations/{conv_id}/messages/stream` | Same, as Server-Sent Events |
| `GET` | `/health` | Liveness check |

See [`docs/IMPLEMENTATION.md#12-api-reference`](docs/IMPLEMENTATION.md#12-api-reference)
for full request/response schemas.

### Tests

```bash
pytest
```

Run from the repo root (`tests/conftest.py` puts `backend/` on `sys.path`
itself, so no `PYTHONPATH` is needed here).

## Running with Docker

`docker-compose.yml` runs the backend and frontend and connects to your
MongoDB Atlas cluster using `MONGO_DB_STRING` from `.env` -- there's no local
MongoDB container:

```bash
cp .env.example .env   # fill in OPENROUTER_API_KEY / GROQ_API_KEY, MONGO_DB_STRING (Atlas), etc
docker compose up --build
```

Ingestion (`parser.py`/`chunker.py`/`embed_and_store.py`) pulls in Docling
plus CPU torch/torchvision, so it's built as a **separate `ingest` profile**
(`backend/Dockerfile.ingest`) and kept out of the default `backend` image.
The containers don't run ingestion automatically; run it once via that
profile, either before or after `docker compose up`:

```bash
docker compose --profile ingest run --rm ingest python -m src.ingest.parser
docker compose --profile ingest run --rm ingest python -m src.ingest.chunker
docker compose --profile ingest run --rm ingest python -m src.ingest.embed_and_store
```

Then the API is at `localhost:8000` and the frontend at `localhost:5173`. If
the backend is reachable at a different host/IP than `localhost`, rebuild the
frontend with `docker compose build --build-arg VITE_API_IP=...` (see the
comment in `docker-compose.yml`).

## Access control model

`DUMMY_USERS` in `backend/src/config/users.py` maps each dummy email to the
`company_id`s it may access, e.g.:

```python
DUMMY_USERS: dict[str, list[str]] = {
    "alice@example.com": ["TCS", "Infosys"],
    "bob@example.com": ["Axis"],
    "carol@example.com": ["Hdfc"],
    "dave@example.com": ["TataTechnologies"],
    "eve@example.com": ["TCS", "Hdfc"],
}
```

Login issues a JWT embedding that company list. `POST /api/query` decodes the
JWT (`get_current_user`) and passes `companies` straight into
`$vectorSearch`'s `filter`, so unauthorized companies' chunks are excluded
from the similarity search itself -- never fetched, never ranked, never
returned.

`POST /api/conversations/{conv_id}/messages` (and its `/stream` sibling) goes
one step further: it only trusts the JWT for identity (the `email` claim)
and re-looks-up that user's authorized companies fresh from `config/users.py`
on every request, rather than trusting the `companies` claim baked into the
token at login time. That way, revoking or changing a user's access takes
effect immediately, even against a JWT that hasn't expired yet.

## Import paths

The codebase mixes two import styles, so a single invocation needs to satisfy
both: `api/main.py` and everything under `api/routes/` use **relative**
imports (`from ..config...`), which only resolve when imported as part of the
`backend.src.*` package (i.e. run from the **repo root**); `retrieval/`,
`ingest/`, `store/`, `graph/`, and `llm/` use **absolute** imports (`from
src.config...`), which only resolve if `backend/` itself is on `sys.path`.
Run everything from the repo root with `PYTHONPATH=backend` set (as in the
commands above) to satisfy both at once -- plain `uvicorn
backend.src.api.main:app` with no `PYTHONPATH` fails with
`ModuleNotFoundError: No module named 'src'`. The Docker image bakes in
`PYTHONPATH=/app/backend` for the same reason.

## Further reading

- [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) -- engineering-reference
  deep dive: every pipeline stage, every LangGraph node, chunking strategy,
  observability, deployment, and known limitations.
- [`docs/system-dossier.html`](docs/system-dossier.html) -- a lighter
  narrative walkthrough of the same system.
