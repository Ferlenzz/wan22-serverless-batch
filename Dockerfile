# БАЗА: dev-образ с CUDA 12.1 и PyTorch 2.4.0
FROM runpod/pytorch:2.4.0-py3.11-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git ffmpeg build-essential ninja-build cmake python3-dev pkg-config \
    libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel packaging ninja

# На всякий: в dev-образе torch уже установлен с CUDA 12.1.
# Если понадобится переустановить, раскомментируй:
# RUN pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0 torchvision==0.19.0

WORKDIR /app
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2

WORKDIR /app/Wan2.2
ENV FORCE_CUDA=1
RUN pip install -r requirements.txt

WORKDIR /app
COPY handler.py /app/handler.py
COPY engine.py  /app/engine.py

ENV RP_VOLUME=/runpod-volume
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-TI2V-5B
ENV BATCH_MAX_SIZE=20
ENV BATCH_LINGER_SEC=5

CMD ["python","-u","/app/handler.py"]
