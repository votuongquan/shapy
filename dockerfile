# Use Python 3.12.10 as base image
FROM python:3.12.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies that might be needed
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to specific version
RUN python -m pip install --upgrade pip==23.3.1

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the entire project
COPY . .

# Install the mesh-mesh-intersection package
RUN pip install /app/mesh-mesh-intersection

# Set Python path environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app/attributes:/usr/local"

# Change to regressor directory and set it as working directory
WORKDIR /app/regressor

# Expose port (adjust as needed for your Flask/web app)
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]