FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    python3-dev \
    git \
    ninja-build \
    hdf5-tools \
    libgl1 \
    libgtk2.0-dev \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg-turbo8-dev \
    libjpeg8-dev \
    libpng-dev \
    libtiff5-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install requirements first
RUN pip install --no-cache-dir -r requirements.txt

# Set CUDA environment variables BEFORE compiling extensions
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set CUDA architectures for broad compatibility
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# Disable ninja for compatibility
ENV MAX_JOBS=1
ENV USE_NINJA=0

# Set PyOpenGL platform for headless rendering
ENV PYOPENGL_PLATFORM=egl

# Clean any existing builds and install
RUN cd /app/mesh-mesh-intersection && \
    rm -rf build/ dist/ *.egg-info/ && \
    pip install . --no-cache-dir --force-reinstall -v

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/attributes:/usr/local"

EXPOSE 8080

CMD ["python", "regressor/app.py"]