# Wan2.2 Serverless (ComfyUI) — файлы и как подготовить workflow

## Что входит
- `Dockerfile` — образ с Torch cu121, ComfyUI, WanVideoWrapper и вашим кодом.
- `engine.py` — запуск Wan 2.2, подстановка параметров в workflow, сохранение результата в `/runpod-volume/results/<job_id>.mp4`.
- `handler.py` — экшены `health`, `warmup`, `gen` (async, без base64), `runsync` (sync, с base64).
- `.runpod/hub.json` и `.runpod/tests.json` — конфиг и тесты.


2) Генерация (sync):
```powershell
$body=@{ input=@{
    action="runsync"; prompt="cinematic portrait, soft rim light";
    width=480; height=832; steps=8; cfg=2.0; length=81;
    user_id="u1"; use_last=$true
}} | ConvertTo-Json -Depth 5
$st = Invoke-RestMethod -Method POST -Uri "https://api.runpod.ai/v2/$ep/runsync" -Headers $hdr -Body $body
$st | Format-List *
```
Готовое видео лежит на томе: `/runpod-volume/results/<job_ts>.mp4`.

## Где лежат веса
В вашем случае weights уже на томе:  
`/runpod-volume/models/Wan2.2-TI2V-5B` (3 shard + index + VAE).  
Пути переданы в контейнер через `WAN_CKPT_DIR` и `WAN_VAE_PATH`.
