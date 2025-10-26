# ---------- BASE ----------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ---------- SYSTEM ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ca-certificates \
    git curl ffmpeg libglib2.0-0 libgl1 libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# ---------- TORCH (cu121) ----------
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0

# XFormers wheel (no-compile) for cu121
RUN pip install xformers==0.0.27.post2 -f https://download.pytorch.org/whl/cu121

# ---------- WAN 2.2 code ----------
WORKDIR /app
ARG WAN_REF=main
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2 && \
    cd /app/Wan2.2 && git fetch --all --tags && git checkout ${WAN_REF}

# убираем flash-attn/xformers из requirements (чтобы не компилились)
WORKDIR /app/Wan2.2
RUN awk 'BEGIN{IGNORECASE=1} !/flash[-_]?attn/ && !/xformers/ {print}' requirements.txt > /tmp/req-pruned.txt && \
    pip install --no-cache-dir --prefer-binary -r /tmp/req-pruned.txt

# ---------- Base runtime deps ----------
RUN pip install --no-cache-dir --prefer-binary \
      runpod==1.6.2 loguru==0.7.2 pillow==10.4.0 \
      imageio==2.36.0 imageio-ffmpeg==0.5.1 numpy==1.26.4 \
      decord==0.6.0 opencv-python-headless==4.10.0.84 \
      safetensors==0.4.5 einops==0.7.0 huggingface_hub==0.20.3

# ---------- Diffusers / Transformers / PEFT ----------
RUN pip install --no-cache-dir --prefer-binary \
      diffusers==0.30.2 transformers==4.44.2 accelerate==0.34.2 \
      peft==0.17.1

# ---------- AUDIO STACK (librosa и зависимости; всё в колёсах) ----------
RUN pip install --no-cache-dir --prefer-binary \
      scipy==1.11.4 \
      numba==0.60.0 llvmlite==0.43.0 \
      scikit-learn==1.3.2 joblib==1.3.2 threadpoolctl==3.2.0 \
      pooch==1.8.2 soundfile==0.12.1 audioread==3.0.1 \
      librosa==0.10.2.post1

# ---------- ПАТЧ №1: фильтрованная загрузка VAE (fallback) ----------
RUN python3 - <<'PY'
from pathlib import Path
import re
p = Path("/app/Wan2.2/wan/modules/vae2_1.py")
s = p.read_text(encoding="utf-8")

helper = r'''
def _load_filtered_state_dict(model, ckpt):
    ms = model.state_dict()
    ok, mism = {}, []
    for k, v in ckpt.items():
        if k in ms and getattr(v, "shape", None) == getattr(ms[k], "shape", None):
            ok[k] = v
        else:
            mism.append(k)
    missing = [k for k in ms.keys() if k not in ok]
    model.load_state_dict(ok, strict=False)
    print(f"[VAE] filtered load: loaded={len(ok)} missing={len(missing)} skipped(mismatch)={len(mism)}")
    return missing, mism
'''
if "def _load_filtered_state_dict" not in s:
    s = s.replace("import torch", "import torch\n" + helper, 1)

# покрываем варианты с/без assign=
s = re.sub(
    r"model\.load_state_dict\(\s*torch\.load\(([^)]+)\)\s*,\s*assign=True\s*\)",
    r"missing, mism = _load_filtered_state_dict(model, torch.load(\1))",
    s, count=1
)
s = re.sub(
    r"model\.load_state_dict\(\s*torch\.load\(([^)]+)\)\s*\)",
    r"missing, mism = _load_filtered_state_dict(model, torch.load(\1))",
    s
)

p.write_text(s, encoding="utf-8")
print("patched (filtered load)", p)
PY

# ---------- ПАТЧ №2: diffusers VAE backend (robust, no strict) ----------
RUN python3 - <<'PY'
from pathlib import Path
p = Path("/app/Wan2.2/wan/modules/vae2_1.py")
s = p.read_text(encoding="utf-8")

append = r'''
# ---- Robust Diffusers VAE backend (no strict state_dict) ----
import os as _os
import torch as _torch
from huggingface_hub import hf_hub_download
import json as _json

try:
    from diffusers import AutoencoderKLWan as _DiffVAE
except Exception:
    from diffusers import AutoencoderKL as _DiffVAE

def _vae_build_diffusers(_device):
    repo_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    token = _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_HUB_TOKEN")
    # 1) тянем config и веса напрямую
    cfg_path = hf_hub_download(repo_id=repo_id, filename="vae/config.json", token=token)
    w_path  = hf_hub_download(repo_id=repo_id, filename="vae/diffusion_pytorch_model.safetensors", token=token)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = _json.load(f)
    # 2) создаём модель по конфигу
    vae = _DiffVAE.from_config(cfg)
    # 3) грузим веса без строгих проверок
    try:
        from safetensors.torch import load_file as _load_sf
        sd = _load_sf(w_path, device="cpu")
    except Exception:
        sd = _torch.load(w_path, map_location="cpu")
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    print(f"[VAE] diffusers load (strict=False): missing={len(missing)} unexpected={len(unexpected)}")
    # 4) перенос на нужное устройство/тип
    vae = vae.to(_device, dtype=_torch.float16 if _device.type == "cuda" else _torch.float32)
    vae.eval().requires_grad_(False)
    return vae

# monkey-patch: при USE_DIFFUSERS_VAE=1 подменяем инициализацию класса
import inspect as _inspect
if _os.environ.get("USE_DIFFUSERS_VAE", "0") == "1":
    print("[VAE] Using diffusers VAE backend (robust)")
    _g = globals()
    # найди класс VAE (имя может меняться между ревизиями)
    _candidates = [k for k,v in _g.items() if _inspect.isclass(v) and "VAE" in k]
    for _name in _candidates:
        _cls = _g.get(_name)
        if not _cls or not hasattr(_cls, "__init__"): 
            continue
        _orig_init = _cls.__init__
        def _init(self, *a, **kw):
            _device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            self.model = _vae_build_diffusers(_device)
        _cls.__init__ = _init
        print("[VAE] patched class:", _name)
        break
'''
if "Robust Diffusers VAE backend" not in s:
    s = s + "\n" + append

p.write_text(s, encoding="utf-8")
print("patched (robust diffusers backend)", p)
PY


# ---------- APP ----------
WORKDIR /app
COPY engine.py /app/engine.py
COPY handler.py /app/handler.py

# ---------- ENV ----------
ENV RP_VOLUME=/runpod-volume
ENV WAN_ROOT=/app/Wan2.2
ENV PYTHONPATH=/app/Wan2.2:$PYTHONPATH
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-TI2V-5B
ENV BATCH_MAX_SIZE=20
ENV BATCH_LINGER_SEC=5
ENV FLASH_ATTENTION_SKIP_COMPILE=1
ENV USE_FLASH_ATTENTION=0
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV MPLCONFIGDIR=/tmp/matplotlib

# переключаемся на diffusers-бэкенд VAE (можно убрать/поставить 0 для отката)
ENV USE_DIFFUSERS_VAE=1

# отключаем «быструю» загрузку HF (чтобы не требовался hf_transfer)
ENV HF_ENABLE_HF_TRANSFER=
ENV HF_HUB_ENABLE_HF_TRANSFER=

# ---------- ENTRY ----------
CMD ["python3","-u","/app/handler.py"]
