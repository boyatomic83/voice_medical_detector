# Demo server for Render (or any container host). Research prototype, not a diagnosis.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    GRADIO_ANALYTICS_ENABLED=False \
    HOST=0.0.0.0 \
    PORT=7860 \
    MODEL_PATH=models/model.joblib

WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY src/ src/
COPY app/ app/
COPY models/model.joblib models/model.joblib

RUN useradd -m appuser
USER appuser
EXPOSE 7860
CMD ["python", "-m", "app.demo"]
