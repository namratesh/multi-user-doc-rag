# frontend

React + TypeScript (Vite) UI for `multi-user-doc-rag`: an email-picker login
screen and a chat screen for the conversational Q&A API. See the [repo-root
README](../README.md) for the system architecture and backend setup -- this
file only covers the frontend.

## Structure

- `src/screens/Login/` -- email-picker login. `demoUsers.ts` mirrors
  `backend/src/config/users.py`'s email -> company map for display purposes
  only; access control is enforced entirely server-side.
- `src/screens/Main/` -- chat screen: conversation `Sidebar`, `ChatPanel` /
  `Message` list, and `ChatInput` for follow-ups.
- `src/api/` -- typed API client (`client.ts`, `types.ts`) and API base URL
  config (`config.ts`, built from `VITE_API_IP` / `VITE_API_PORT`).
- `src/auth/useAuth.ts` -- holds the JWT returned by `/api/auth/login` and
  attaches it to subsequent requests.
- `src/components/` -- shared UI (`Button`, `Stamp`, `Wordmark`).

## Setup

```bash
cp .env.example .env   # VITE_API_IP / VITE_API_PORT -- must be reachable from the browser
npm install
npm run dev
```

Open the printed Vite URL (default `http://localhost:5173`), pick one of the
demo emails on the login screen, and chat. The backend must already be
running (see repo-root README) -- `VITE_API_IP`/`VITE_API_PORT` are baked
into the client's API base URL at build/dev time.

## Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Vite dev server with HMR |
| `npm run build` | Type-check (`tsc -b`) then production build to `dist/` |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | Oxlint |

## Docker

`frontend/Dockerfile` builds the static bundle and serves it via nginx
(`nginx.conf`). `VITE_API_IP`/`VITE_API_PORT` are build args, not runtime
env vars -- they get baked into the compiled JS, so they must point at a
host the **browser** can reach (e.g. the backend's published port), not an
in-network Docker service name. When running via the repo-root
`docker-compose.yml`, rebuild with
`docker compose build --build-arg VITE_API_IP=... frontend` if the backend
isn't reachable at `localhost:8000` from the browser.
