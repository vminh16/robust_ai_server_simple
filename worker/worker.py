import os
import json
import numpy as np 
import redis
import signal
import time 
import torch
import torch.nn as nn

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
QUEUE_KEY = os.getenv("QUEUE_KEY", "task_queue")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/linear_model.pth")

stop = False

def handle_sigterm(signum, frame):
    global stop
    print("Received SIGTERM, shutting down gracefully...")
    stop = True

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)


def main():
    print ("Worker started")
    print ('Model path:', MODEL_PATH)
    state = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    w = state['weight']
    input_dim = w.size(1)
    output_dim = w.size(0)
    model = nn.Linear(input_dim, output_dim)
    model.load_state_dict(state)
    model.eval()

    print ("Model loaded successfully\nConnecting to Redis...")
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    print ("Connected to Redis, waiting for tasks...")

    while not stop:
        try:
            item = r.blpop(QUEUE_KEY, timeout=2)
            if item:
                _, task_data = item
                task = json.loads(task_data)
                print(f"Received task: {task['id']} with data: {task['data']}")
                input_data = np.array(task['data'], dtype=np.float32)
                input_tensor = torch.from_numpy(input_data).unsqueeze(0)

                # Make prediction
                with torch.no_grad():
                    output = model(input_tensor)
                result = output.squeeze().item()
                print(f"Predicted result for task {task['id']}: {result}")

                # Store result in Redis with task ID as key
                task_id = task.get("task_id", task.get("id"))  # tạm tương thích cả 2
                result_key = f"result:{task_id}"
                r.setex(result_key, 600, json.dumps({"prediction": result}))


        except Exception as e:
            print(f"Error processing task: {e}")
            time.sleep(1)
    print("Worker shutting down gracefully.")


if __name__ == "__main__":
    main()