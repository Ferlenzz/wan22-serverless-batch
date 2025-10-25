import os, json, base64, subprocess, tempfile, shutil
from pathlib import Path
from typing import List, Dict

TMP_ROOT = Path(os.environ.get("TMPDIR","/tmp"))
TMP_ROOT.mkdir(parents=True, exist_ok=True)

def _b64_video(path: Path) -> str:
    raw = path.read_bytes()
    return "data:video/mp4;base64," + base64.b64encode(raw).decode()

def _extract_last_frame(video_path: Path, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg","-y","-i",str(video_path),"-vf","select=eq(n\\,last)","-vframes","1",str(out_png)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class _Engine:
    def __init__(self):
        # TODO: разовая инициализация WAN 2.2 / Comfy графа
        pass

    # ---- ВСТАВЬ СВОЙ РЕАЛЬНЫЙ ВЫЗОВ WAN 2.2 ЗДЕСЬ ----
    def _run_wan22_one(self, job: Dict, tmp_dir: Path) -> Path:
        """
        Сгенерируй видео и верни путь к mp4 в tmp_dir.
        Здесь — заглушка через ffmpeg (крутит 1 кадр), замени на свой вызов.
        """
        vid = tmp_dir / "vid.mp4"
        png = tmp_dir / "f.png"
        # если клиент передал исходное изображение
        img_b64 = (job.get("image_base64") or "")
        if "," in img_b64: img_b64 = img_b64.split(",",1)[1]
        if img_b64:
            png.write_bytes(base64.b64decode(img_b64))
        else:
            from PIL import Image
            w=int(job.get("width",480)); h=int(job.get("height",832))
            Image.new("RGB",(w,h),(10,10,10)).save(png)
        # NVENC-кодек для скорости/цены
        cmd = ["ffmpeg","-y","-loop","1","-i",str(png),"-t","1.0","-r","8",
               "-c:v","h264_nvenc","-preset","p4","-qp","23", str(vid)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return vid
    # ----------------------------------------------------

    def process_batch(self, batch: List[Dict], results_dir: Path, last_dir: Path) -> List[Dict]:
        out=[]
        results_dir.mkdir(parents=True, exist_ok=True)
        last_dir.mkdir(parents=True, exist_ok=True)

        for job in batch:
            jid = job["job_id"]; uid = job.get("user_id","u1")
            tmp = TMP_ROOT / f"wan22_{jid}"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)
            try:
                vid = self._run_wan22_one(job, tmp)

                # last frame
                last_png = last_dir / f"{uid}.png"
                try: _extract_last_frame(vid, last_png)
                except Exception: pass

                final = results_dir / f"{jid}.mp4"
                shutil.move(str(vid), final)

                payload = {
                    "status": "COMPLETED",
                    "job_id": jid,
                    "video_path": str(final),
                    # "video_base64": _b64_video(final),  # включай только для runsync, иначе трафик
                }
                (results_dir / f"{jid}.json").write_text(json.dumps(payload, ensure_ascii=False))
                out.append(payload)
            except Exception as e:
                payload = {"status":"FAILED","job_id":jid,"error":str(e)}
                (results_dir / f"{jid}.json").write_text(json.dumps(payload, ensure_ascii=False))
                out.append(payload)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        return out

ENGINE = _Engine()
