FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ARG SERVICE=api
ENV SERVICE=${SERVICE}
COPY requirements*.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ "$SERVICE" = "dashboard" ]; then \
        pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-dashboard.txt; \
    else \
        pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt; \
    fi

COPY . .
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 CMD if [ "$SERVICE" = "dashboard" ]; then curl -fsS http://127.0.0.1:8501/_stcore/health; else curl -fsS http://127.0.0.1:8000/health; fi || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
