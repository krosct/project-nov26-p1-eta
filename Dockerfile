FROM python:3.12-slim

# LightGBM precisa da biblioteca OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

# Modelo gerado por scripts/train.py (não versionado)
COPY models ./models

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "project_nov26_p1_eta.api:app", "--host", "0.0.0.0", "--port", "8000"]
