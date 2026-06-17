# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG OCAML_COMPILER=ocaml-base-compiler.5.4.1

ENV OPAMYES=1 \
    OPAMROOT=/opt/opam \
    PATH=/opt/opam/default/bin:/opt/uv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/loom-venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      clang \
      curl \
      gcc \
      g++ \
      git \
      jq \
      make \
      m4 \
      ocaml \
      opam \
      patch \
      pkg-config \
      python3.12 \
      python3.12-dev \
      python3.12-venv \
      rsync \
      sudo \
      unzip \
      xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/uv/bin \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uv/bin sh \
    && uv --version

RUN opam init --disable-sandboxing --bare \
    && opam switch create default "${OCAML_COMPILER}" \
    && opam install dune.3.22.0 yojson

WORKDIR /workspace

FROM base AS python-deps

# Cache Python dependencies in the image when only project dependency metadata changes.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

FROM python-deps AS runtime

COPY . .
RUN opam exec -- dune build src/loom_cli/main.exe \
    && uv run --no-sync python -m py_compile experiments/source/harness/run.py

ENTRYPOINT ["/workspace/scripts/bench/run-in-container.sh"]
