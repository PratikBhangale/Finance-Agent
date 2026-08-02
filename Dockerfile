# --------------------------------------------------------------------------
# Finance Agent — multi-stage Dockerfile using uv as the package manager
# --------------------------------------------------------------------------
# Build:   docker build -t finance-agent .
# Run:     docker run -p 8000:8000 --env-file finance_agent/.env finance-agent
# --------------------------------------------------------------------------

# ---- Stage 1: install dependencies into a virtual env --------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation for faster cold starts
ENV UV_COMPILE_BYTECODE=1
# Use the system Python rather than downloading a managed one
ENV UV_PYTHON_PREFERENCE=only-system

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (without the project itself yet) into .venv
RUN uv sync --frozen --no-install-project --no-dev

# Now copy the full source and install the project
COPY . .
RUN uv sync --frozen --no-dev


# ---- Stage 2: lean runtime image -----------------------------------------
FROM python:3.13-slim-bookworm AS runtime

WORKDIR /app

# Copy the entire app including the pre-built .venv from the builder
COPY --from=builder /app /app

# Put the virtual env's bin directory on PATH so `uvicorn` is found directly
ENV PATH="/app/.venv/bin:$PATH"

# Cloud Run sets PORT; default to 8000 for local docker run
ENV PORT=8000

EXPOSE ${PORT}

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Run with uvicorn; use $PORT so Cloud Run can inject its own port
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
