# Multi-stage-friendly, lean image for the multi-agent supply-chain API.
FROM python:3.12-slim

# system deps sometimes needed by prophet/statsmodels wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- dependency layer first (cached unless requirements change) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- then the application code ---
COPY src/ ./src/

# data is mounted at runtime (not baked into the image) — keeps the image lean
# and stateless. Mount your Rossmann CSVs to /app/data.
VOLUME ["/app/data"]

EXPOSE 8000

# GROQ_API_KEY can be passed at run time to enable LLM routing/synthesis;
# without it the service falls back to keyword routing + template reports.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
