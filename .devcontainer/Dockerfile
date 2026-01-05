# Copy vllm from nvidia vllm image
# Use intermediate stage to handle the wildcard in dist-info name
FROM nvcr.io/nvidia/vllm:25.12.post1-py3 AS vllm-source

FROM nvcr.io/nvidia/pytorch:25.12-py3

# Copy vllm package and CLI from nvidia image
COPY --from=vllm-source /usr/local/lib/python3.12/dist-packages/vllm /usr/local/lib/python3.12/dist-packages/vllm
COPY --from=vllm-source /usr/local/bin/vllm /usr/local/bin/vllm

# Copy dist-info (version varies, so we use mount + shell glob)
RUN --mount=from=vllm-source,source=/usr/local/lib/python3.12/dist-packages,target=/vllm-src \
    cp -r /vllm-src/vllm-*.dist-info /usr/local/lib/python3.12/dist-packages/

# Install vllm runtime dependencies (not included in pytorch base image)
# Versions pinned from nvcr.io/nvidia/vllm:25.12.post1-py3
# Already in pytorch:25.12-py3 (same version): numba, nvidia-ml-py, prometheus_client,
#   python-json-logger, pyzmq, tornado
RUN pip install --no-cache-dir \
    blake3==1.0.8 \
    cachetools==6.2.4 \
    cbor2==5.7.1 \
    cloudpickle==3.1.2 \
    compressed-tensors==0.12.2 \
    depyf==0.20.0 \
    diskcache==5.6.3 \
    gguf==0.17.1 \
    lark==1.2.2 \
    llguidance==1.3.0 \
    lm-format-enforcer==0.11.3 \
    "mistral_common[audio,image]==1.8.8" \
    msgspec==0.20.0 \
    openai==2.14.0 \
    openai-harmony==0.0.8 \
    opencv-python-headless==4.11.0.86 \
    outlines_core==0.2.11 \
    partial-json-parser==0.2.1.1.post7 \
    prometheus-fastapi-instrumentator==7.1.0 \
    py-cpuinfo==9.0.0 \
    pybase64==1.4.3 \
    "ray[default]==2.53.0" \
    sentencepiece==0.2.1 \
    setproctitle==1.3.7 \
    tiktoken==0.12.0 \
    watchfiles==1.1.1 \
    xgrammar==0.1.27

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      git git-lfs build-essential ninja-build cmake python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Make sure CUDA is visible to compilers, and include user local bin for pip-installed scripts
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/home/ubuntu/.local/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Target Blackwell GB10. Build native cubins for SM120 & SM121 and include PTX.
# Many PyTorch extensions read TORCH_CUDA_ARCH_LIST.
# MAX_JOBS=8 to avoid OOM during CUDA compilation
ENV TORCH_CUDA_ARCH_LIST="12.0;12.1+PTX" \
    MAX_JOBS=8 \
    FORCE_CUDA=1

# Optional: if a project uses CMake's CUDA archs
ENV CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120;121"

# Triton with Blackwell support
RUN pip install --no-cache-dir --upgrade triton==3.5.1

# Build causal-conv1d and mamba-ssm from source (hardcode architectures, no env var control)
RUN pip install --no-cache-dir --no-build-isolation \
    "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d@v1.5.4" \
    mamba-ssm==2.2.6.post3

# Build flash-attn from source (GB10 sm_121 only for faster builds)
# RUN FLASH_ATTN_CUDA_ARCHS="121" \
#     pip install --no-cache-dir --no-build-isolation \
#     flash-attn==2.8.3

# Build flash-linear-attention from source (for KDA support)
RUN pip install --no-cache-dir --no-build-isolation \
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention@67eee20c8503cd19eeb52aa1b99821308e9260c5"

# Grant sudo to existing ubuntu user (UID/GID 1000 matches host)
RUN apt-get update \
    && apt-get install -y sudo \
    && echo "ubuntu ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Set SSH agent socket for forwarded SSH keys (used by devcontainer mount)
ENV SSH_AUTH_SOCK=/ssh-agent

# Install Claude Code CLI for ubuntu user (bump version to trigger reinstall)
ARG CLAUDE_CODE_VERSION=2.0.74
USER ubuntu
RUN curl -fsSL https://claude.ai/install.sh | bash -s ${CLAUDE_CODE_VERSION}
USER root

# Note: Fast-LLM itself is installed at container creation time via postCreateCommand
# This allows for editable installs with live code changes
