# Declare global build argument for compute capability (cpu or gpu)
ARG DEVICE="cpu"

# ==============================================================================
# STAGE 1: Runtime Base (Lean & Secure)
# ==============================================================================
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install only basic shared libraries needed for execution (OpenCV/FFmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user for runtime safety
RUN useradd -m -u 1000 worker
WORKDIR /app

# Ensure we include the virtual environment in the PATH
ENV PATH="/app/.venv/bin:$PATH"

# ==============================================================================
# STAGE 2: Builder (Temporary compilation & package installation)
# ==============================================================================
FROM base AS builder

# Install build tools, compiler, curl, and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# ==============================================================================
# STAGE 3: Production Builder (Prepare production dependencies)
# ==============================================================================
FROM builder AS production-builder
ARG DEVICE

COPY pyproject.toml uv.lock README.md ./

# Sync only third-party production dependencies, deferring workspace/project install
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra headless --extra ${DEVICE} --frozen --no-dev --no-install-project

# Copy source code and register the project
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra headless --extra ${DEVICE} --frozen --no-dev

# Strip debug symbols and prune caches/tests to reduce image footprint
RUN find /app/.venv -name "*.so" -exec strip --strip-unneeded {} + && \
    find /app/.venv -name "__pycache__" -type d -exec rm -rf {} + && \
    find /app/.venv -type d -name "tests" -exec rm -rf {} +

# ==============================================================================
# STAGE 4: Test Builder (Prepare development/test dependencies)
# ==============================================================================
FROM builder AS test-builder
ARG DEVICE

COPY pyproject.toml uv.lock README.md ./

# Sync all dependencies (including dev), deferring workspace/project install
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra headless --extra ${DEVICE} --frozen --no-install-project

# Copy source code, tests, and register the project
COPY src/ src/
COPY tests/ tests/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra headless --extra ${DEVICE} --frozen

# Strip debug symbols and prune caches to reduce image footprint
RUN find /app/.venv -name "*.so" -exec strip --strip-unneeded {} + && \
    find /app/.venv -name "__pycache__" -type d -exec rm -rf {} +

# ==============================================================================
# STAGE 5: Final Production Runner (Minimal & Hardened)
# ==============================================================================
FROM base AS production

# Copy only the compiled and pruned virtual environment from the production builder
COPY --chown=worker:worker --from=production-builder /app/.venv /app/.venv

# Copy source code and config files
COPY --chown=worker:worker src/ src/
COPY --chown=worker:worker system_config.docker_example.yml system_config.yml
COPY --chown=worker:worker pyproject.toml uv.lock README.md ./

# Switch to non-root user
USER worker

ENTRYPOINT ["python", "-m", "loitering_detector.scripts.main", "run"]

# ==============================================================================
# STAGE 6: Final Test Runner
# ==============================================================================
FROM base AS test

# Copy the complete compiled and pruned virtual environment from the test builder
COPY --chown=worker:worker --from=test-builder /app/.venv /app/.venv

# Copy source code, tests, and configs
COPY --chown=worker:worker src/ src/
COPY --chown=worker:worker tests/ tests/
COPY --chown=worker:worker system_config.docker_example.yml system_config.yml
COPY --chown=worker:worker pyproject.toml uv.lock README.md ./

# Switch to non-root user
USER worker

ENTRYPOINT ["pytest"]
