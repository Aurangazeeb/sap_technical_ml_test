# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Copy only what pip needs to build the package
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Build and install the package + all runtime deps into a prefix directory.
# pip fetches the uv_build backend declared in [build-system] automatically.
RUN pip install --no-cache-dir --prefix=/install .


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN groupadd --system appgroup && useradd --system --gid appgroup appuser

WORKDIR /app

# Bring installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy data at build time (override at runtime via SAP_DATA_PATH mount if needed)
COPY data/ ./data/

# Create a writable cache directory and hand ownership to appuser.
# Must happen before USER appuser; --system users have no home dir, so we
# redirect HuggingFace / sentence-transformers cache to /app/.cache instead.
RUN mkdir -p /app/.cache && chown -R appuser:appgroup /app

# Configuration — override any value via environment variables (SAP_ prefix)
ENV SAP_DATA_PATH=data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson
ENV SAP_API_HOST=0.0.0.0
ENV SAP_API_PORT=8000
# Redirect HuggingFace model cache into /app/.cache (writable by appuser)
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8000

USER appuser

CMD ["uvicorn", "sap_cxii_tech_ex_01.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
