FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y git ffmpeg wget python3 python3-pip python3-venv libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip wheel
RUN pip install runpod==1.6.1 numpy==1.26.4 torch==2.4.0 torchvision==0.19.0

WORKDIR /app
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2
WORKDIR /app/Wan2.2
RUN pip install -r requirements.txt

WORKDIR /app
COPY handler.py /app/handler.py
COPY engine.py  /app/engine.py

ENV RP_VOLUME=/runpod-volume
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-TI2V-5B
ENV BATCH_MAX_SIZE=20
ENV BATCH_LINGER_SEC=5

CMD ["python","-u","/app/handler.py"]
