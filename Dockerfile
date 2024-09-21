# Start with CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV, Eigen, and PCL dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libeigen3-dev \
    libpcl-dev \
    libflann-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Set the entrypoint
ENTRYPOINT ["./build/cuda_mode_hpr"]