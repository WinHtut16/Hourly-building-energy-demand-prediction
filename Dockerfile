# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps (LightGBM needs libgomp) ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (layer caching) ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application files ────────────────────────────────────────────────────
COPY app.py .
COPY model.pkl .
COPY templates/ templates/

# ── Render injects PORT at runtime; default to 10000 ──────────────────────────
ENV PORT=10000

# ── Run with Gunicorn (production WSGI server) ────────────────────────────────
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
