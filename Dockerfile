# ---------------------------------------------------------------
# CensusInsight PK - production image
# Build:  docker build -t censusinsight-pk .
# Run:    docker run -p 8501:8501 censusinsight-pk
#         (optional) -e OPENAI_API_KEY=sk-...  to enable the LLM agent
# ---------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8501 \
    MPLBACKEND=Agg

WORKDIR /app

# fonts for Matplotlib charts in the PDF report
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Rebuild the clean dataset + boundaries from the raw PBS tables and pre-train
# the ML model at build time, so the container starts instantly.
RUN python scripts/build_dataset.py && python scripts/build_geo.py && python scripts/train_model.py

# run as non-root user (required by Hugging Face Spaces, good practice everywhere)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fs http://localhost:${PORT}/_stcore/health || exit 1

CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0"]
