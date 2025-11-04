FROM ultralytics/ultralytics:latest-jetson-jetpack6


ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Copy project files
COPY . /app

# Update and install essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git wget cmake build-essential \
    curl ca-certificates \
    python3-setuptools python3-numpy libopencv-dev \
    python3-opencv \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
    libgtk2.0-dev pkg-config libjpeg-dev libpng-dev libtiff-dev \
    libgtk-3-dev  libgl1-mesa-dev    libglib2.0-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Upgrade pip
RUN pip3 install --upgrade pip

RUN pip3 uninstall -y opencv-python opencv-python-headless

# Install Python packages
RUN pip3 install \
        Pillow \
        matplotlib \
        tqdm \
        pyyaml \
        requests \
        pypylon

#Expose RTSP Port
EXPOSE 8000/udp

# Default command
CMD ["python3", "detect.py"]
