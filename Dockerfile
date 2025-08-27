# 🤖 Next-Generation Hive-Mind Lottery Prediction System
# Multi-stage Docker build for optimized production deployment

# Build stage - Python dependencies and model preparation
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/
COPY config/ ./config/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p logs models/pytorch models/sklearn

# Production stage - Minimal runtime image
FROM python:3.11-slim as production

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r hive && useradd -r -g hive hive

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application files
COPY --from=builder --chown=hive:hive /app/ /app/

# Create necessary directories with proper permissions
RUN mkdir -p logs data models config && \
    chown -R hive:hive /app

# Switch to non-root user
USER hive

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV HIVE_MIND_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/monitoring/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "src/main.py"]