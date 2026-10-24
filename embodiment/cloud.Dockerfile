FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git build-essential python3 python3-venv ffmpeg libgl1 libegl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libvulkan1 libsm6 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/bootstrap && /opt/bootstrap/bin/pip install --no-cache-dir uv==0.13.4
ENV PATH=/opt/bootstrap/bin:$PATH UV_CACHE_DIR=/workspace/viper/cache/uv OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAX_JOBS=2 CUDA_HOME=/usr/local/cuda NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
ENV HF_HOME=/workspace/viper/cache/huggingface XDG_CACHE_HOME=/workspace/viper/cache/xdg TORCH_HOME=/workspace/viper/cache/torch TORCH_EXTENSIONS_DIR=/workspace/viper/cache/torch_extensions TRITON_CACHE_DIR=/workspace/viper/cache/triton CUDA_CACHE_PATH=/workspace/viper/cache/cuda
WORKDIR /opt/viper
COPY . /opt/viper
ARG INSTALL_RUNTIME=0
RUN if [ "$INSTALL_RUNTIME" = "1" ]; then /opt/bootstrap/bin/python -m embodiment.cloud_run --prefix /opt/viper-runtime --setup both --execute; fi
ENTRYPOINT ["/opt/bootstrap/bin/python", "-m", "embodiment.cloud_run"]
