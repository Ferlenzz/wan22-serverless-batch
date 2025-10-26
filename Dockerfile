# ---------- BASE ----------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ---------- SYSTEM ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ca-certificates \
    git curl ffmpeg libglib2.0-0 libgl1 libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# ---------- TORCH (cu121) ----------
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0

# XFormers (готовое колесо под cu121)
RUN pip install xformers==0.0.27.post2 -f https://download.pytorch.org/whl/cu121

# ---------- WAN 2.2 ----------
WORKDIR /app
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2

WORKDIR /app/Wan2.2
# Вырезаем flash-attn/xformers из requirements, чтобы ничего не компилировалось
RUN awk 'BEGIN{IGNORECASE=1} !/flash[-_]?attn/ && !/xformers/ {print}' requirements.txt > /tmp/req-pruned.txt
RUN pip install -r /tmp/req-pruned.txt

# Перевод на VAE 2.2 в image2video
RUN sed -i 's/from \.modules\.vae2_1 import Wan2_1_VAE/from .modules.vae2_2 import Wan2_2_VAE/' /app/Wan2.2/wan/image2video.py \
 && sed -i 's/self\.vae = Wan2_1_VAE/self.vae = Wan2_2_VAE/' /app/Wan2.2/wan/image2video.py

# ---------- НАШ КОД ----------
WORKDIR /app
COPY handler.py /app/handler.py
COPY engine.py  /app/engine.py
COPY .runpod/tests.json /app/.runpod/tests.json
COPY .runpod/hub.json   /app/.runpod/hub.json

# ---------- PY DEPS (SDK/утилиты) ----------
# Важно: peft >= 0.17.0. Ставим 0.17.1 (0.18.0 недоступен в текущем индексе).
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
      pillow>=10 \
      "imageio[ffmpeg]>=2.34" \
      numpy>=1.26 \
      loguru>=0.7 \
      runpod==1.7.13 \
      einops>=0.7.0 \
      librosa==0.10.2.post1 \
      soundfile==0.12.1 \
      decord>=0.6.0 \
      diffusers==0.30.2 \
      peft==0.17.1

# ---------- ENV ----------
ENV RP_VOLUME=/runpod-volume
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-TI2V-5B
ENV BATCH_MAX_SIZE=20
ENV BATCH_LINGER_SEC=5
ENV FLASH_ATTENTION_SKIP_COMPILE=1
ENV USE_FLASH_ATTENTION=0
# кэш HF на общий том — ускоряет холодный старт
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV MPLCONFIGDIR=/tmp/matplotlib

# ---------- ENTRY ----------
CMD ["python3","-u","/app/handler.py"]
