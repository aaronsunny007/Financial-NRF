# Results-explorer demo (no model, no GPU). Build from the repo root:
#   docker build -t nrf-demo .
#   docker run -p 7860:7860 nrf-demo
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=7860

WORKDIR /app

COPY deploy/requirements-docker.txt deploy/requirements-docker.txt
RUN pip install --no-cache-dir -r deploy/requirements-docker.txt

COPY deploy/ deploy/
COPY results/ results/

# Run as an unprivileged user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["python", "deploy/app.py"]
