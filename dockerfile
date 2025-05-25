FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

WORKDIR /app

# Cài Python 3.10 và các công cụ cần thiết
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.10 python3.10-dev python3.10-distutils python3-pip build-essential hdf5-tools libgl1 libgtk2.0-dev

# Upgrade pip cho Python 3.10
RUN python3.10 -m pip install --upgrade pip==23.3.1

# SOLUTION 1: Force ignore installed packages and use --break-system-packages
# RUN python3.10 -m pip install --break-system-packages --force-reinstall blinker

# SOLUTION 2: Remove system blinker package first (RECOMMENDED)
RUN apt-get remove -y python3-blinker || true

# Cài PyTorch, torchvision, torchaudio cho CUDA 12.4 và Python 3.10
RUN python3.10 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

COPY . /app

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.9"

# Install requirements with ignore-installed for problematic packages
RUN python3.10 -m pip install --no-cache-dir --ignore-installed blinker -r requirements.txt

# Cài gói mesh-mesh-intersection
RUN python3.10 -m pip install /app/mesh-mesh-intersection

# Set Python path environment variables
ENV PYTHONPATH="/app/shapy/attributes:/usr/local"

# Set working directory to regressor folder
WORKDIR /app/regressor

# Expose port (adjust if your app uses a different port)
EXPOSE 8080

CMD ["python3.10", "app.py"]