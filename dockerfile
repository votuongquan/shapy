FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    python3-dev \
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

RUN python -m pip install --upgrade pip==23.3.1
RUN pip install --no-cache-dir -r requirements.txt

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=CUDAHOME/bin:PATH
ENV LD_LIBRARY_PATH=CUDAHOME/lib64:LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# Set PyOpenGL platform for headless rendering
ENV PYOPENGL_PLATFORM=egl

RUN pip install /app/mesh-mesh-intersection

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/attributes:/usr/local"

EXPOSE 8080

CMD ["python", "regressor/app.py"]