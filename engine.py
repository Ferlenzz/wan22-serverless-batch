# engine.py
import os, json, base64, subprocess, tempfile, shutil
from pathlib import Path
from typing import List, Dict

# Если в окружении задан свой кеш моделей – используем
HF_HOME = os.environ.get("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Папка, куда писать временные и итоговые файлы
TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _save_base64_video(video_path: Path) -> str:
    raw = video_path.read_bytes()
    return "data:video/mp4;base64," + base64.b64encode(raw).decode()


def _extract_last_frame(video_path: Path, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # последний кадр (быстро и дёшево, NVDEC)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", "select=eq(n\\,last)", "-vframes", "1",
        str(out_png)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class _Engine:
    """
    Обёртка над фактическим вызовом Wan 2.2/Comfy.
    Здесь оставлен «крюк» WAN 2.2 CALL, чтобы ты поставил свой реальный вызов.
    Все остальные обязательства: батч, сохранение результатов, last image – здесь.
    """

    def __init__(self):
        # Подними здесь все тяжёлые зависимости один раз (модель, пайплайн и т.д.)
        # self.pipe = your_wan22_pipeline(...)
        pass

    # -----------------------------
    # Реальный вызов Wan 2.2
    # -----------------------------
    def _run_wan22_one(self, job: Dict, tmp_dir: Path) -> Path:
        """
        Верни путь к mp4 для одного job. Здесь должен быть твой реальный вызов модели.
        Я оставляю stub, который ожидает, что ты заменишь его на свой вызов Comfy/Wan:
          - читаешь job['prompt'], job['image_base64'], job['lora_pairs'], width/height/steps/cfg/length
          - генеришь видео (mp4) в tmp_dir/vid.mp4
        """
        vid = tmp_dir / "vid.mp4"

        # --------- WAN 2.2 CALL (замени на свой код) ----------
        # Пример: subprocess на твой comfy-workflow / python-entrypoint
        # Здесь просто чтобы не падало, создадим 1-кадровое mp4.
        # У тебя этот участок должен генерить полноценный ролик.
        png = tmp_dir / "f.png"
        png.write_bytes(base64.b64decode((job.get("image_base64") or "").split(",")[-1] or b""))
        if not png.exists() or png.stat().st_size == 0:
            # если начального изображения нет – сделаем пустую заглушку
            from PIL import Image
            Image.new("RGB", (job.get("width", 480), job.get("height", 832)), (10, 10, 10)).save(png)

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(png),
            "-t", "1.0", "-r", "8",
            "-c:v", "h264_nvenc", "-preset", "p4", "-qp", "23",
            str(vid),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # ------------------------------------------------------

        return vid

    # -----------------------------
    # Batch wrapper
    # -----------------------------
    def process_batch(self, batch: List[Dict], results_dir: Path, last_dir: Path) -> List[Dict]:
        results = []
        results_dir.mkdir(parents=True, exist_ok=True)

        for job in batch:
            job_id  = job["job_id"]
            user_id = job.get("user_id", "u1")
            tmp = TMP_ROOT / f"wan22_{job_id}"
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)

            try:
                video_path = self._run_wan22_one(job, tmp)

                # положим last frame
                last_png = last_dir / f"{user_id}.png"
                try:
                    _extract_last_frame(video_path, last_png)
                except Exception:
                    pass

                # сохраним итоговый mp4 на сети-том
                final_mp4 = results_dir / f"{job_id}.mp4"
                shutil.move(str(video_path), final_mp4)

                # при runsync можно вернуть base64, но по умолчанию лучше не слать его (тяжело)
                res = {
                    "status": "COMPLETED",
                    "job_id": job_id,
                    "video_path": str(final_mp4),
                    # "video_base64": _save_base64_video(final_mp4),  # ← если нужно прямо в ответе
                }
                (results_dir / f"{job_id}.json").write_text(json.dumps(res, ensure_ascii=False))
            except Exception as e:
                res = {
                    "status": "FAILED",
                    "job_id": job_id,
                    "error": str(e),
                }
                (results_dir / f"{job_id}.json").write_text(json.dumps(res, ensure_ascii=False))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

            results.append(res)

        return results


ENGINE = _Engine()
