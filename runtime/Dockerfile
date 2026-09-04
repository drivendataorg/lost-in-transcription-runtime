## BASE IMAGE
FROM nvidia/cuda:13.0.3-runtime-ubuntu22.04 AS base
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    SHELL=/bin/bash

# Install system dependencies
COPY apt.txt /tmp/apt.txt
RUN apt-get update --fix-missing \
    && apt-get update \
    && xargs -a /tmp/apt.txt apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/apt.txt

# Create runtime user
ENV RUNTIME_USER=runtimeuser
ENV RUNTIME_UID=1000
ENV RUNTIME_GID=1000

RUN echo "Creating ${RUNTIME_USER} user..." \
    && groupadd --gid ${RUNTIME_GID} ${RUNTIME_USER} \
    && useradd --create-home --gid ${RUNTIME_GID} --no-log-init --uid ${RUNTIME_UID} ${RUNTIME_USER}

# Set up code execution working directory
RUN mkdir /code_execution && \
    chown -R ${RUNTIME_USER}:${RUNTIME_USER} /code_execution
WORKDIR /code_execution

# Switch to runtime user for Python package installation
USER ${RUNTIME_USER}

# Copy files for environment
COPY --chown=${RUNTIME_USER}:${RUNTIME_USER} \
    .python-version \
    pyproject.toml \
    uv.lock \
    ./

# uv environment variables
ENV UV_NO_DEV=1 \
    UV_FROZEN=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies
# Use cache mount to speed up rebuilds
ENV UV_CACHE_DIR=/home/runtimeuser/.cache/uv
ENV UV_PYTHON_CACHE_DIR=/home/runtimeuser/.cache/uv/python
# kenlm fixes its highest n-gram order when it compiles, and its default of 6
# rejects the order 7 to 10 models that character-level LMs often use.
ENV MAX_ORDER=10
# uv keys a built wheel on the source archive alone, so it ignores MAX_ORDER
# and the compression libraries in apt.txt. Raise the number on the cache id
# whenever either of those changes. A new id starts an empty cache, so kenlm
# gets built again instead of restored from a wheel that predates the change.
RUN --mount=type=cache,id=uv-cache-1,target=/home/runtimeuser/.cache/uv,uid=1000,gid=1000 \
    uv sync

# Copy entrypoint
COPY --chown=${RUNTIME_USER}:${RUNTIME_USER} \
    entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

## RUNTIME
FROM base AS runtime
ENV UV_NO_SYNC=1
# The runtime base has no C++/CUDA toolchain, so flashinfer's JIT
# compile fails at first use. Force vllm to the Triton kernels,
# which ship prebuilt inside the wheel.
ENV VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    VLLM_USE_FLASHINFER_SAMPLER=0
CMD ["bash", "/entrypoint.sh"]

## TEST-BASE
FROM base AS test-base
ENV UV_PYTHON_CACHE_DIR=/home/runtimeuser/.cache/uv/python
RUN --mount=type=cache,id=uv-cache-1,target=/home/runtimeuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra test
COPY --chown=${RUNTIME_USER}:${RUNTIME_USER} tests/ ./tests/

## TEST-RUNTIME
FROM test-base AS test-runtime
ENV UV_NO_SYNC=1
CMD ["bash", "-c", "uv run pytest tests"]
