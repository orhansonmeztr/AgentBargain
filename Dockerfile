# ============================================================
# AgentBargain - Multi-stage Docker Build
# ============================================================
# Uses uv for fast, reproducible dependency installation.
# ============================================================

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies (no dev deps in production)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./
COPY README.md ./

# COPY keeps the host's file modes, and the app runs as an unprivileged user.
# A source file that happened to be 0600 on the build machine became
# unreadable in the image and crashed the import at startup. Normalise the
# modes here; the runtime stage copies them as they are now.
RUN chmod -R a+rX src alembic alembic.ini README.md

# Install the project itself
RUN uv sync --frozen --no-dev

# Stage 2: Production image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Copy source, migrations, and static files
COPY --from=builder /app/src /app/src
COPY --from=builder /app/alembic /app/alembic
COPY --from=builder /app/alembic.ini /app/alembic.ini

# Create data directory for SQLite and ChromaDB
RUN mkdir -p /app/data

# Run unprivileged. The image only needs to read /app and write /app/data.
# A real home directory is required: ChromaDB writes its cache/settings under
# $HOME, and without one every index operation dies with PermissionError —
# which silently aborts discovery indexing for the whole negotiation.
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --home-dir /home/appuser --create-home appuser \
    && chown -R appuser:appuser /app/data
USER appuser

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
# A container must listen on all interfaces to be reachable through its port
# mapping. The app refuses to start like this without API_KEYS, unless
# ALLOW_UNAUTHENTICATED=true is set explicitly.
ENV SERVER_HOST=0.0.0.0

# Expose port
EXPOSE 8000

# Health check.
# raise_for_status() matters: a bare httpx.get() exits 0 even on a 500, so the
# container would have reported "healthy" while the API was erroring.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5).raise_for_status()" || exit 1

# Run migrations and then start the application
CMD ["sh", "-c", "alembic upgrade head && agent-bargain"]
