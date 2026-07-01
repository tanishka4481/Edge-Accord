# Stage 1: Build Environment
FROM python:3.11-slim AS build-env

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install PlatformIO
RUN pip install --no-cache-dir platformio

# Pre-install PlatformIO platform toolchains and library dependencies to cache them
WORKDIR /workspace
COPY firmware/platformio.ini /workspace/firmware/
RUN pio pkg install --project-dir /workspace/firmware

# Stage 2: Full Runner Environment (adds Node.js, @wokwi/cli, and Python deps)
FROM build-env AS runner

# Install Node.js & npm (for @wokwi/cli)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @wokwi/cli \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy all project files into the workspace
COPY . /workspace/

# Default behavior: run bash
CMD ["bash"]
