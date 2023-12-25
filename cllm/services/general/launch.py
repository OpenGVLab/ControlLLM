import argparse
from .tools import *
from fastapi import Form, Body
from fastapi.responses import JSONResponse
from cllm.services import app, pool
import uvicorn


parser = argparse.ArgumentParser(description="Image Perception API")
parser.add_argument("--host", type=str, default="localhost", help="Host")
parser.add_argument("--port", type=int, default=10049, help="Port")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
args = parser.parse_args()


@app.post("/remote_logging")
@pool.register(lambda: Logger(args.device))
async def remote_logging(
    history_msgs: list = Body(...),
    task_decomposition: list = Body(...),
    solution: list = Body(...),
    record: str = Form(...),
    like: bool = Form(...),
):
    model = remote_logging.__wrapped__.model
    output = model(history_msgs, task_decomposition, solution, record, like)
    return JSONResponse(output)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
