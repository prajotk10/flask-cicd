# ===========================
# Stage 1: Build Dependencies
# ===========================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies needed for compiling some packages
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies globally in builder stage
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# ===========================
# Stage 2: Runtime Image
# ===========================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages and binaries from builder
COPY --from=builder /install /usr/local

COPY . .

# Create non-root user and set permissions
RUN useradd -m -u 1000 flaskuser \
    && chown -R flaskuser:flaskuser /app

USER flaskuser

# Expose port 5000
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "main:app"]
