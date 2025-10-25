# 1) DEV-образ с CUDA 12.1 + PyTorch 2.4 (есть компилятор/SDK)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git ffmpeg build-essential ninja-build cmake python3-dev pkg-config \
    libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel packaging ninja cmake

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

RUN pip install --no-cache-dir --no-build-isolation \
    flash-attn==2.5.8 \
    -f https://flash-attn.s3.amazonaws.com/whl/torch-2.4.0_cu121.html

RUN pip install --no-cache-dir \
    xformers==0.0.27.post2 \
    -f https://download.pytorch.org/whl/cu121

WORKDIR /app
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2

WORKDIR /app/Wan2.2
RUN sed -i '/flash-attn/d;/xformers/d' requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY handler.py /app/handler.py
COPY engine.py  /app/engine.py

# 9) ENV
ENV RP_VOLUME=/runpod-volume
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-TI2V-5B
ENV BATCH_MAX_SIZE=20
ENV BATCH_LINGER_SEC=5

CMD ["python","-u","/app/handler.py"]
