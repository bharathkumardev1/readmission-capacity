# Scoring service. The model files come from `readmit run` and are mounted at /models.
#
#   docker build -t readmit .
#   docker run --rm -p 8000:8000 -v "$PWD/artifacts:/models:ro" readmit
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 \
    READMIT_MODEL_DIR=/models READMIT_CAPACITY=0.10

# LightGBM's wheel needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock .
RUN pip install -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

RUN useradd --create-home --uid 1000 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["uvicorn", "readmit.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
