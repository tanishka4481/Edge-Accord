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

# Stage 2: Full Runner Environment (adds Wokwi CLI & Python deps)
FROM build-env AS runner

# Download wokwi-cli precompiled Linux binary directly (raw executable file)
RUN curl -L -o /usr/local/bin/wokwi-cli https://github.com/wokwi/wokwi-cli/releases/latest/download/wokwi-cli-linux-x64 \
    && chmod +x /usr/local/bin/wokwi-cli

# Install Python requirements
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy all project files into the workspace
COPY . /workspace/

# Default behavior: run bash
CMD ["bash"]