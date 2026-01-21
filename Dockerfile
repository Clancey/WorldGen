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

# Install PyTorch3D from source
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# Install all other dependencies first (before UniK3D)
RUN pip install \
    "diffusers>=0.33.1" \
    "xformers>=0.0.30" \
    "transformers>=4.48.3" \
    "py360convert>=0.1.0" \
    "einops>=0.7.0" \
    "pillow>=8.0.0" \
    "scikit-image>=0.24.0" \
    "sentencepiece>=0.2.0" \
    "peft>=0.7.1" \
    "open3d>=0.19.0" \
    "trimesh>=4.6.1" \
    "opencv-python" \
    "flask"

# Install nunchaku (prebuilt wheel for Python 3.11 Linux)
RUN pip install "nunchaku @ https://github.com/mit-han-lab/nunchaku/releases/download/v0.2.0/nunchaku-0.2.0+torch2.7-cp311-cp311-linux_x86_64.whl"

# Install viser fork
RUN pip install "git+https://github.com/ZiYang-xie/viser.git"

# Clone and install UniK3D from source with correct CUDA settings
RUN git clone --depth 1 https://github.com/lpiccinelli-eth/UniK3D.git /tmp/UniK3D && \
    cd /tmp/UniK3D && \
    pip install --no-build-isolation . && \
    rm -rf /tmp/UniK3D

# Copy and install main package (dependencies already installed)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-deps .

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
