FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git git-lfs wget curl build-essential cmake ninja-build \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libusb-1.0-0 libglfw3 libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Set CUDA architecture for building extensions (RTX 3090 = SM_86)
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV FORCE_CUDA=1

# Install PyTorch first
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install PyTorch3D from source with correct CUDA arch
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# Clone and install UniK3D separately with proper build isolation disabled
RUN git clone --depth 1 https://github.com/lpiccinelli-eth/UniK3D.git /tmp/UniK3D && \
    cd /tmp/UniK3D && \
    pip install --no-build-isolation -e . && \
    cd /app

# Copy and install main package (without UniK3D since it's already installed)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install remaining dependencies (UniK3D already installed)
RUN pip install . --no-deps || pip install .

# Install Flask for web UI
RUN pip install flask

COPY demo.py ./

# Clone and install ml-sharp (optional feature)
RUN mkdir -p submodules && \
    git clone --depth 1 https://github.com/apple/ml-sharp.git submodules/ml-sharp && \
    pip install -e submodules/ml-sharp

# Copy web UI
COPY web/ ./web/

RUN mkdir -p /app/output /root/.cache/huggingface

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 5000 8080

CMD ["python", "web/app.py"]
