# BIZRA DDAGI OS — Production Frontend Dockerfile
# Multi-stage build: Node 22 (build) → Nginx 1.27 Alpine (serve)
#
# Build context: frontend/
# CI usage:  docker build -t bizra-frontend .        (from frontend/)
# Compose:   docker compose up frontend              (from repo root)
#
# Standing on Giants: Vite · Nginx Alpine · Docker BuildKit
# Constitutional Constraint: Ihsan >= 0.95, bundle < 512KB

# =============================================================================
# Stage 1: Builder — install deps + production build
# =============================================================================
FROM node:22-alpine AS builder

WORKDIR /build

# Copy package manifests first (layer cache: only reinstalls when deps change)
COPY package.json package-lock.json ./

# Install all dependencies (dev deps needed for tsc + vite build)
RUN npm ci --no-audit --no-fund

# Copy source files (order: least → most frequently changing)
COPY tsconfig.json tsconfig.node.json ./
COPY index.html vite.config.ts ./
COPY public/ public/
COPY src/ src/

# Build args injected at CI time for traceability
ARG VERSION=dev
ARG BUILD_DATE=unknown
ARG GIT_SHA=unknown

# Production build (TypeScript compile + Vite bundle)
RUN npm run build

# =============================================================================
# Stage 2: Runtime — minimal Nginx image serving static assets
# =============================================================================
FROM nginx:1.27-alpine AS runtime

LABEL org.opencontainers.image.title="BIZRA DDAGI OS Frontend"
LABEL org.opencontainers.image.description="Sovereign decentralized OS interface"
LABEL org.opencontainers.image.version="0.3.0"
LABEL org.opencontainers.image.vendor="BIZRA"
LABEL org.opencontainers.image.source="https://github.com/BizraInfo/bizra-data-lake"

# Remove default Nginx welcome page
RUN rm -rf /usr/share/nginx/html/*

# Copy Nginx config as envsubst template
# The official nginx image auto-processes /etc/nginx/templates/*.template
# at startup, resolving ${API_UPSTREAM} and ${WS_UPSTREAM} from environment.
COPY nginx.conf /etc/nginx/templates/default.conf.template

# Copy built assets from builder stage
COPY --from=builder /build/dist /usr/share/nginx/html

# Make runtime dirs writable by nginx user (required for envsubst + pid + cache)
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /etc/nginx/conf.d && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown -R nginx:nginx /var/run/nginx.pid

USER nginx

# Runtime-configurable upstream targets (no secrets baked in)
ENV API_UPSTREAM=python-api:8000 \
    WS_UPSTREAM=desktop-bridge:9743

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:8080/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]
