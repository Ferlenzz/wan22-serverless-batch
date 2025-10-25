# Лёгкая база: CUDA 12.1 + cuDNN8 (runtime)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ca-certificates \
    git ffmpeg libglib2.0-0 libgl1 curl \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# PyTorch 2.4.0 + cu121 (готовые колёса)
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0

# Только xformers (готовое колесо под cu121), БЕЗ flash-attn
RUN pip install \
    xformers==0.0.27.post2 \
    -f https://download.pytorch.org/whl/cu121

# Клонируем Wan2.2
WORKDIR /app
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2

# Убираем любые упоминания flash-attn/xformers из requirements, чтобы не пересобирались
WORKDIR /app/Wan2.2
RUN sed -i '/flash-attn/d;/xformers/d' requirements.txt

# Ставим остальные зависимости Wan2.2
RUN pip install -r requirements.txt

# Наш серверлесс-код
WORKDIR /app
COPY handler.py /app/handler.py
COPY engine.py  /app/engine.py

# На всякий: выключаем любые попытки автосборки flash-attn
ENV FLASH_ATTENTION_SKIP_COMPILE=1
ENV USE_FLASH_ATTENTION=0

# ENV рантайма
ENV RP_VOLUME=/runpod-volume
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-TI2V-5B
ENV BATCH_MAX_SIZE=20
ENV BATCH_LINGER_SEC=5

CMD ["python3","-u","/app/handler.py"]
