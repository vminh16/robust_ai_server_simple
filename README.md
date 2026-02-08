
# Robust AI Server (FastAPI + Redis + Worker)

## Services
- api: FastAPI
- redis: redis:alpine
- worker: Python worker consuming `task_queue` and writing `result:{task_id}`

## Run
```bash
docker compose up --build
=======
# robust_ai_server_simple
