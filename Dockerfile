# WorldGen Docker Image
# Requires NVIDIA GPU with CUDA support

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python to not buffer output
ENV PYTHONUNBUFFERED=1

# Set up locale
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libusb-1.0-0 \
    libglfw3 \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install PyTorch with CUDA support first
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install PyTorch3D (built from source)
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# Install the main package and dependencies
RUN pip install .

# Copy demo script and other files
COPY demo.py ./

# Copy submodules if they exist (for ml-sharp support)
COPY submodules/ ./submodules/ 2>/dev/null || true

# Install ml-sharp if available (optional)
RUN if [ -d "submodules/ml-sharp" ]; then \
    pip install -e submodules/ml-sharp; \
    fi

# Create directories for output and model cache
RUN mkdir -p /app/output /root/.cache/huggingface

# Set Hugging Face cache directory
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Expose port for Viser visualization
EXPOSE 8080

# Default command runs the demo
ENTRYPOINT ["python", "demo.py"]
CMD ["--help"]
