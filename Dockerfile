# База — существует и лёгкая: CUDA 12.1 + cuDNN8 (runtime)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Меньше слоёв, без мусора
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ca-certificates \
    git ffmpeg libglib2.0-0 libgl1 curl \
 && rm -rf /var/lib/apt/lists/*

# pip без кеша — экономим место
ENV PIP_NO_CACHE_DIR=1
RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# PyTorch 2.4.0 + cu121 (готовые официальные колёса)
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0

# Готовые бинарные whl для flash-attn и xformers под torch2.4/cu121
RUN pip install --no-build-isolation \
    flash-attn==2.5.8 \
    -f https://flash-attn.s3.amazonaws.com/whl/torch-2.4.0_cu121.html
RUN pip install \
    xformers==0.0.27.post2 \
    -f https://download.pytorch.org/whl/cu121

# Клонируем Wan2.2 и ставим зависимости (НЕ переустанавливаем flash-attn/xformers)
WORKDIR /app
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2
WORKDIR /app/Wan2.2
RUN sed -i '/flash-attn/d;/xformers/d' requirements.txt
RUN pip install -r requirements.txt

# Кладём наш серверлесс-код
WORKDIR /app
COPY handler.py /app/handler.py
COPY engine.py  /app/engine.py

# ENV
ENV RP_VOLUME=/runpod-volume
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-TI2V-5B
ENV BATCH_MAX_SIZE=20
ENV BATCH_LINGER_SEC=5

CMD ["python3","-u","/app/handler.py"]
