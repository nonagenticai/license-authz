ARG BASE_REGISTRY=docker.io

# Build stage
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS builder

WORKDIR /app

# uv pinned the same way as licenses/secrets/billing-service (multi-arch binary)
COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /usr/local/bin/uv

# Use the image's own python so the venv interpreter path is valid at runtime
ENV UV_PYTHON_DOWNLOADS=never

# Copy manifests first for cache. uv.lock is REQUIRED: installs are frozen to it.
COPY pyproject.toml uv.lock ./

# Install dependencies only, exactly as locked (incident #61: an unlocked
# `uv pip install .` resolved pyjwt 2.15.1 while uv.lock pinned 2.14.0)
RUN uv sync --frozen --no-dev --no-install-project

# Copy project files
COPY src/ src/

# Install the local project (non-editable, as before; deps already locked)
RUN uv sync --frozen --no-dev --no-editable

# Build-time guard: fail the build if any installed version != uv.lock
# (pyjwt asserted explicitly). The script stays in the builder stage only.
COPY scripts/assert_lock_versions.py /tmp/assert_lock_versions.py
RUN uv export --frozen --no-dev --no-emit-project --no-hashes -o /tmp/lock-pins.txt && \
    /app/.venv/bin/python /tmp/assert_lock_versions.py uv.lock /tmp/lock-pins.txt && \
    uv sync --frozen --no-dev --no-editable --check

# Runtime stage
FROM ${BASE_REGISTRY}/library/python:3.14-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Switch to non-root user (uid 1000, created above)
# hadolint ignore=DL3066
USER appuser

# Expose port
EXPOSE 9001

# Health check
# hadolint ignore=DL3025
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:9001/health', timeout=2).raise_for_status()"

# Run the service.
# --timeout-keep-alive 75: uvicorn's default (5s) is shorter than the upstream
# idle timeout of the istio-ingressgateway's pooled connections (envoy default
# 1h; no DestinationRule sets one for this host). Envoy then reuses a connection
# uvicorn has just closed and the licence-check Lua httpCall fails with
# `503 lua_response` ("reset reason: connection termination") without the
# request ever reaching this service (seen 2026-10-15 02:19:59Z, 02:51:28Z).
# The server must be the side that holds idle connections LONGER.
# Guarded by tests/test_uvicorn_keep_alive.py.
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9001", "--timeout-keep-alive", "75"]
