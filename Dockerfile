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

# Ставим xformers (готовое колесо под cu121). FLASH-ATTN НЕ СТАВИМ.
RUN pip install xformers==0.0.27.post2 -f https://download.pytorch.org/whl/cu121

# Клонируем Wan2.2
WORKDIR /app
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2

# --------- ВАЖНО: жёстко вырезаем flash-attn/xformers из любых requirements ---------
WORKDIR /app/Wan2.2
# Создадим "пропиленный" requirements, куда НЕ попадут строки с flash(-|_)attn или xformers
RUN awk 'BEGIN{IGNORECASE=1} !/flash[-_]?attn/ && !/xformers/ {print}' requirements.txt > /tmp/req-pruned.txt

# Ставим зависимости из очищенного файла
RUN pip install -r /tmp/req-pruned.txt
# --------------------------------------------------------------------

# Наш серверлесс-код
WORKDIR /app
COPY handler.py /app/handler.py
COPY engine.py  /app/engine.py

# На всякий: запретим любые автосборки flash-attn
ENV FLASH_ATTENTION_SKIP_COMPILE=1
ENV USE_FLASH_ATTENTION=0

# ENV рантайма
ENV RP_VOLUME=/runpod-volume
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-TI2V-5B
ENV BATCH_MAX_SIZE=20
ENV BATCH_LINGER_SEC=5

# SDK RunPod + полезные штуки для логов
RUN pip install --no-cache-dir runpod==1.6.6 loguru

# (не обязательно, но удобно если используешь FastAPI/uvicorn)
# RUN pip install --no-cache-dir fastapi uvicorn

# чтобы логи шли сразу
ENV PYTHONUNBUFFERED=1

CMD ["python3","-u","/app/handler.py"]
