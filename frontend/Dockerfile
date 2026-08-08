# Build context is frontend/ itself (see docker-compose.yml).
#
# VITE_API_IP/VITE_API_PORT are baked into the static bundle at build time
# (Vite resolves import.meta.env.VITE_* at build time, not runtime), so they
# must be the address the *browser* can reach the backend at -- typically the
# host machine's published backend port (e.g. localhost:8000), not the
# in-network Docker service name ("backend"), since the fetch calls run in
# the user's browser, outside the compose network.
FROM node:22-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_API_IP=localhost
ARG VITE_API_PORT=8000
ENV VITE_API_IP=${VITE_API_IP}
ENV VITE_API_PORT=${VITE_API_PORT}

RUN npm run build

FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
