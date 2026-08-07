# multi-user-doc-rag

Multi-user document search & conversational Q&A over a handful of companies'
earnings-call transcripts, with per-user access control: each user is scoped
to the company (or companies) their dummy account is authorized for, enforced
at the vector-search layer itself.

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

## Setup

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

Fill in `.env`:

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | embeddings + (future) chat completions |
| `SECRET_KEY` | JWT signing secret |
| `MONGO_DB_STRING` (or `MONGODB_URI`) | Atlas connection string, password filled in |
| `MONGODB_DB_NAME` / `MONGODB_COLLECTION` / `MONGODB_VECTOR_INDEX` | optional, sensible defaults |

### 3. Install & ingest

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd backend
python -m src.ingest.parser        # data/*.pdf -> data/parsed/*.json
python -m src.ingest.chunker       # data/parsed/*.json -> data/chunks/*.json
python -m src.ingest.embed_and_store   # embeds chunks, upserts into MongoDB Atlas,
                                         # creates the vector index on first run
```

The vector index takes ~30-60s to become queryable after first creation;
`embed_and_store.py` waits for that automatically.

### 4. Run the API

```bash
uvicorn backend.src.api.main:app --reload
```

```bash
# Login as a dummy user (see backend/src/config/users.py for the email -> company map)
curl -s -X POST localhost:8000/api/auth/login -H 'content-type: application/json' \
  -d '{"email": "alice@example.com"}' | tee /tmp/login.json

TOKEN=$(python3 -c "import json;print(json.load(open('/tmp/login.json'))['access_token'])")

# Query -- results are restricted to alice's authorized companies (TCS, Infosys)
curl -s -X POST localhost:8000/api/query -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"query": "What was revenue growth?"}'
```

## Access control model

`DUMMY_USERS` in `backend/src/config/users.py` maps each dummy email to the
`company_id`s it may access, e.g.:

```python
DUMMY_USERS = {
    "alice@example.com": ["TCS", "Infosys"],
    "bob@example.com": ["Axis"],
    "eve@example.com": ["TCS", "Hdfc"],
}
```

Login issues a JWT embedding that company list. `POST /api/query` decodes the
JWT (`get_current_user`) and passes `companies` straight into
`$vectorSearch`'s `filter`, so unauthorized companies' chunks are excluded
from the similarity search itself -- never fetched, never ranked, never
returned.
