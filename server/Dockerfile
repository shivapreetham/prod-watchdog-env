FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY models.py .
COPY server/ ./server/
COPY openenv.yaml .

# HF Spaces listens on port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
