# База: готовый PyTorch 2.4 + CUDA 12.1 (runtime) — компактнее, чем cuda:devel
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

# Минимальные системные зависимости
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git ffmpeg libglib2.0-0 libgl1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Инструменты pip
RUN python -m pip install --upgrade pip setuptools wheel packaging

# Устанавливаем ГОТОВЫЕ бинарные колёса (без сборки) под torch==2.4/cu121
# flash-attn
RUN pip install --no-cache-dir --no-build-isolation \
    flash-attn==2.5.8 \
    -f https://flash-attn.s3.amazonaws.com/whl/torch-2.4.0_cu121.html
# xformers
RUN pip install --no-cache-dir \
    xformers==0.0.27.post2 \
    -f https://download.pytorch.org/whl/cu121

# Клонируем Wan2.2
WORKDIR /app
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2

# Уберём строки flash-attn/xformers из requirements, чтобы не переустанавливало
WORKDIR /app/Wan2.2
RUN sed -i '/flash-attn/d;/xformers/d' requirements.txt

# Ставим остальные зависимости проекта
RUN pip install --no-cache-dir -r requirements.txt

# Наш серверлесс-код
WORKDIR /app
COPY handler.py /app/handler.py
COPY engine.py  /app/engine.py

# ENV
ENV RP_VOLUME=/runpod-volume
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-TI2V-5B
ENV BATCH_MAX_SIZE=20
ENV BATCH_LINGER_SEC=5

CMD ["python","-u","/app/handler.py"]
