# Wan2.2 Serverless (ComfyUI) — файлы и как подготовить workflow

## Что входит
- `Dockerfile` — образ с Torch cu121, ComfyUI, WanVideoWrapper и вашим кодом.
- `engine.py` — запуск ComfyUI headless, подстановка параметров в workflow, сохранение результата в `/runpod-volume/results/<job_id>.mp4`.
- `handler.py` — экшены `health`, `warmup`, `gen` (async, без base64), `runsync` (sync, с base64).
- `.runpod/hub.json` и `.runpod/tests.json` — конфиг и тесты.
- `workflows/` — пример места, где должен лежать `new_Wan22_api.json` **на томе** (см. ниже).

## Подготовка workflow (самое важное)
Нужен рабочий JSON ComfyUI для Wan2.2 (например, `new_Wan22_api.json` из вашего репозитория).
Файл должен оказаться **на сетевом томе** по пути:
```
/runpod-volume/workflows/new_Wan22_api.json
```

Самый быстрый способ — через временный Pod:

1. RunPod → Pods → Deploy Pod (любой Ubuntu/PyTorch), подключи свой Network Volume с mount path `/runpod-volume`.
2. Открой Terminal и выполни:
   ```bash
   mkdir -p /runpod-volume/workflows
   # вариант 1: загрузить с локальной машины через S3 (если у тома есть S3 API)
   # вариант 2: вставить содержимое через редактор nano/vi
   nano /runpod-volume/workflows/new_Wan22_api.json
   # вставь JSON, сохрани
   ```
3. Проверь, что файл на месте:
   ```bash
   ls -lah /runpod-volume/workflows/new_Wan22_api.json
   ```

> Важно: внутри workflow у узлов должны быть поля, которые мы подставляем из кода:  
  `prompt`, `width`, `height`, `steps`, `cfg`, `length`, `image_base64`, `model_dir`, `vae_path`.  
  Если имена отличаются — поправьте ключи в `engine.py` в функции `_set_input(...)` (словарь `slots`).

## Быстрый тест
1) Прогрев:
```powershell
$ep="YOUR_ENDPOINT_ID"; $key="YOUR_API_KEY"
$hdr=@{Authorization="Bearer $key"; "Content-Type"="application/json"}
$body=@{ input=@{ action="warmup" } } | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri "https://api.runpod.ai/v2/$ep/runsync" -Headers $hdr -Body $body
```

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
