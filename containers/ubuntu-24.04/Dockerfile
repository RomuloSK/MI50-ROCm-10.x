ARG BASE_IMAGE=ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGET=gfx906
ARG RUST_TOOLCHAIN=1.98.0

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    THEROCK_AMDGPU_FAMILIES=${TARGET} \
    THEROCK_DIST_AMDGPU_FAMILIES=${TARGET} \
    THEROCK_TEST_AMDGPU_FAMILIES=${TARGET}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bash \
       autoconf \
       automake \
       bison \
    build-essential \
    ca-certificates \
       ccache \
       clang \
       cmake \
       curl \
       file \
       flex \
       gfortran \
       git \
       git-lfs \
       libegl-dev \
       libelf-dev \
       libgl-dev \
       libglx-dev \
       libtool \
       libx11-dev \
       python3-mako \
       libnuma-dev \
       libsqlite3-dev \
       ninja-build \
       pkg-config \
       python3 \
       python3-dev \
       python3-magic \
       python3-yaml \
       python3-pip \
       python3-venv \
       rsync \
       rustup \
       tcl \
       texinfo \
       unzip \
       xz-utils \
       zstd \
    && rustup toolchain install "${RUST_TOOLCHAIN}" --profile minimal \
    && rustup default "${RUST_TOOLCHAIN}" \
    && git lfs install --system \
    && python3 -m pip install --no-cache-dir --break-system-packages \
       CppHeaderParser==2.7.4 \
       joblib==1.5.1 \
       msgpack==1.1.1 \
       zstandard==0.25.0 \
       meson==1.4.2 \
       pytest==8.4.2 \
       pytest-subtests==0.15.0 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}" \
    RUSTUP_TOOLCHAIN=${RUST_TOOLCHAIN}

WORKDIR /workspace
COPY sources.lock.json support-matrix.json README.md /workspace/
COPY scripts /workspace/scripts

CMD ["bash"]
