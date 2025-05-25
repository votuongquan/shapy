FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
WORKDIR /app

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    python3-dev \
    hdf5-tools \
    libgl1 \
    libgtk2.0-dev


COPY . /app

RUN python -m pip install --upgrade pip==23.3.1
RUN pip install --no-cache-dir -r requirements.txt

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.9"

RUN pip install /app/mesh-mesh-intersection

EXPOSE 8080

CMD ["python", "regressor/app.py"]