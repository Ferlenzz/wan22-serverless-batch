import os, base64, subprocess, tempfile, shlex
from pathlib import Path

RP_VOLUME  = Path(os.environ.get("RP_VOLUME", "/runpod-volume"))
WAN_CKPT   = Path(os.environ.get("WAN_CKPT_DIR", "/runpod-volume/models/Wan2.2-TI2V-5B"))
WAN_REPO   = Path("/app/Wan2.2")
PYTHON_BIN = "python"

class WanEngine:
    _initialized = False
    def __init__(self):
        self._init_model()

    def _init_model(self):
        if self._initialized: return
        assert WAN_CKPT.exists(), f"Checkpoint dir not found: {WAN_CKPT}"
        self._initialized = True

    def _tmp_from_b64(self, data_url: str, name="img.png") -> Path:
        data = data_url.split(",")[-1]
        raw  = base64.b64decode(data)
        p = Path(tempfile.mkdtemp())/name
        p.write_bytes(raw)
        return p

    def _choose_inputs(self, task: dict):
        user_id = task.get("user_id","anon")
        last_path = None
        if task.get("last_image_base64"):
            last_path = self._tmp_from_b64(task["last_image_base64"], "last.png")
        elif task.get("last_image_path"):
            p = Path(task["last_image_path"]);  last_path = p if p.exists() else None
        elif bool(task.get("reuse_last_image", False)):
            cand = RP_VOLUME/"state"/"last_images"/f"{user_id}.png"
            if cand.exists(): last_path = cand

        if task.get("image_base64"):
            start = self._tmp_from_b64(task["image_base64"], "start.png")
        elif task.get("image_path"):
            start = Path(task["image_path"])
        else:
            raise ValueError("start image required (image_base64|image_path)")
        return start, last_path

    def generate_one(self, task: dict) -> dict:
        job_id  = task["job_id"]
        user_id = task.get("user_id","anon")
        start_img, last_img = self._choose_inputs(task)

        prompt = task["prompt"]
        width  = int(task.get("width", 480))
        height = int(task.get("height", 832))
        length = int(task.get("length", 64))
        steps  = int(task.get("steps", 8))
        cfg    = float(task.get("cfg", 2.0))

        out_mp4 = Path(tempfile.mkdtemp())/f"{job_id}.mp4"
        use_img = last_img or start_img

        cmd = f"""{PYTHON_BIN} generate.py --task ti2v-5B --size {width}*{height} \
          --ckpt_dir {shlex.quote(str(WAN_CKPT))} --offload_model True --convert_model_dtype --t5_cpu \
          --prompt {shlex.quote(prompt)} --image_path {shlex.quote(str(use_img))} \
          --steps {steps} --length {length} --cfg {cfg} --output {shlex.quote(str(out_mp4))}"""
        subprocess.run(cmd, shell=True, check=True, cwd=str(WAN_REPO))

        last_png = RP_VOLUME/"state"/"last_images"/f"{user_id}.png"
        last_png.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg","-y","-sseof","-0.1","-i", str(out_mp4), "-vframes","1", str(last_png)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        b64 = base64.b64encode(out_mp4.read_bytes()).decode()
        return {"job_id": job_id, "status":"COMPLETED", "video": f"data:video/mp4;base64,{b64}"}

ENGINE = WanEngine()
