# Stage 1: Base image with system dependencies
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Optimization: System dependencies and cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Security: Create non-root user
RUN useradd -m -u 1000 worker
WORKDIR /app

# Install uv globally
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# Stage 2: Production build (headless)
FROM base AS production

COPY pyproject.toml uv.lock README.md ./
# Caching: Use uv cache mount for faster builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra headless --frozen --no-dev

COPY src/ src/
COPY system_config.docker_example.yml system_config.yml

# Security: Switch to non-root user
RUN chown -R worker:worker /app
USER worker

# Entrypoint: Point directly to refactored run script
ENTRYPOINT ["uv", "run", "python", "-m", "loitering_detector.scripts.main", "run"]

# Stage 3: Test build (includes dev dependencies and tests)
FROM base AS test

COPY pyproject.toml uv.lock README.md ./
# Caching: Use uv cache mount for faster builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra headless --frozen

COPY src/ src/
COPY tests/ tests/
COPY system_config.docker_example.yml system_config.yml

# Security: Switch to non-root user
RUN chown -R worker:worker /app
USER worker

ENTRYPOINT ["uv", "run", "pytest"]
