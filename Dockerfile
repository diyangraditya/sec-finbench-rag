# Multi-stage, high-performance Python 3.12 container using official uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PORT=8000

WORKDIR /app

# Install system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies in a separate layer for optimal caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source code
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY main.py ./

# Place virtual environment binaries directly into PATH
ENV PATH="/app/.venv/bin:$PATH"

# Default port exposure: 8000 (FastAPI) and 8501 (Streamlit)
EXPOSE 8000 8501

# Default command starts FastAPI backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
