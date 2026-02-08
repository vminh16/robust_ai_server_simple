from fastapi import FastAPI 
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
import os
import json
import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
QUEUE_KEY = os.getenv("QUEUE_KEY", "task_queue")

app = FastAPI(title = "Robust AI Server - API")

r : Optional[redis.Redis] = None

class PredictRequest(BaseModel):
   data: List[float] = Field(..., description="Input data for prediction")

# Endpoint startup event to initialize Redis connection
@app.on_event("startup")
async def start_up():
    global r
    r = redis.from_url(REDIS_URL, decode_responses=True)

@app.on_event("shutdown")
async def shut_down():
    global r
    if r is not None:
        await r.close()

# Endpoint to handle prediction requests
@app.post("/predict")
async def predict(request: PredictRequest):
    task_id = str(uuid.uuid4())
    task_data = {
        "id": task_id,
        "data": request.data
    }
    assert r is not None, "Redis connection not initialized"
    await r.rpush(QUEUE_KEY, json.dumps(task_data))
    return {"message": "Prediction task queued", "task_id": task_id}

# Endpoint to get the results of a prediction task
@app.get("/result/{task_id}")
async def get_result(task_id: str):
    assert r is not None, "Redis connection not initialized"
    result = await r.get(f"result:{task_id}")
    if result is None:
        return {"message": "Result Processing", "task_id": task_id}
    return {"task_id": task_id, "result": json.loads(result)}
