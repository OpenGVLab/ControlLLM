import argparse

import uvicorn
from typing import Any, Dict, AnyStr, List, Union

from cllm.services import app, pool
from .llama2 import *

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

parser = argparse.ArgumentParser(description="LLAMA2 API")
parser.add_argument("--host", type=str, default="localhost", help="Host")
parser.add_argument(
    "--model",
    type=str,
    default="/mnt/afs/share_data/tianhao2/llama2/Llama-2-13b-chat-hf",
    help="model path",
)
parser.add_argument("--port", type=int, default=10051, help="Port")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
args = parser.parse_args()


@app.post("/llama2_chat")
@pool.register(lambda: LLaMABot(args.device, args.model))
async def llama2_chat(messages: JSONStructure = None):
    model = llama2_chat.__wrapped__.model
    output = model(messages)
    return output


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
