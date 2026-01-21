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

# Copy and install
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
RUN pip install .

# Install Flask for web UI
RUN pip install flask

COPY demo.py ./
COPY submodules/ ./submodules/ 2>/dev/null || true
RUN if [ -d "submodules/ml-sharp" ]; then pip install -e submodules/ml-sharp; fi

# Copy web UI
COPY web/ ./web/

RUN mkdir -p /app/output /root/.cache/huggingface

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 5000 8080

CMD ["python", "web/app.py"]
