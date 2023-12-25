import argparse
from PIL import Image
import io
import uvicorn

from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse

from .tools import *
from cllm.services import app, pool

parser = argparse.ArgumentParser(description="VQA API")
parser.add_argument("--host", type=str, default="localhost", help="Host")
parser.add_argument("--port", type=int, default=10049, help="Port")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
args = parser.parse_args()


@app.post("/vilt_qa")
@pool.register(lambda: Vilt(args.device))
async def vilt_qa(image: UploadFile = File(None), text: str = Form(...)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = vilt_qa.__wrapped__.model
    output = model(image, text)
    return JSONResponse(output)


@app.post("/llava")
@pool.register(lambda: LLaVA(args.device))
async def llava(image: UploadFile = File(None), text: str = Form(...)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = llava.__wrapped__.model
    output = model(image, text)
    return JSONResponse(output)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
