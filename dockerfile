# Multi-stage build for smaller final image
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy project files
COPY . .

# Upgrade pip
RUN python -m pip install --upgrade pip==23.3.1

# Install requirements and mesh-mesh-intersection
RUN pip install -r requirements.txt
RUN pip install ./shapy/mesh-mesh-intersection

# Final stage - runtime image
FROM python:3.12-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
WORKDIR /app
COPY . .

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/attributes:/usr/local"

# Set working directory to regressor
WORKDIR /app/regressor

# Default command
CMD ["python", "app.py"]