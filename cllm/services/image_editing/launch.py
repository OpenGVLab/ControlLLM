import argparse

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from PIL import Image
import io

from .tools import *
from cllm.services import app, pool

parser = argparse.ArgumentParser(description="Image Editing API")
parser.add_argument("--host", type=str, default="localhost", help="Host")
parser.add_argument("--port", type=int, default=10049, help="Port")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
args = parser.parse_args()


def ImageResponse(image):
    img_stream = io.BytesIO()
    image.save(img_stream, format="png")
    img_stream.seek(0)
    return StreamingResponse(img_stream, media_type="image/png")


@app.post("/instruct_pix2pix")
@pool.register(lambda: InstructPix2Pix(args.device))
async def instruct_pix2pix(image: UploadFile = File(None), text: str = Form(...)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = instruct_pix2pix.__wrapped__.model
    output = model(image, text)
    return ImageResponse(output)


@app.post("/partial_image_editing")
@pool.register(lambda: PartialImageEditing(args.device))
async def partial_image_editing(
    image: UploadFile = File(None),
    mask: UploadFile = File(None),
    prompt: str = Form(...),
):
    print(f"image: {image}; \n\nmask: {mask}; \n\nprompt: {prompt}")
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    mask_bytes = mask.file.read()
    mask = Image.open(io.BytesIO(mask_bytes))
    model = partial_image_editing.__wrapped__.model
    output = model(image, mask, prompt)
    return ImageResponse(output)


@app.post("/inpainting_ldm")
@pool.register(lambda: LDMInpainting(args.device))
async def inpainting_ldm(image: UploadFile = File(None), mask: UploadFile = File(None)):
    image = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    mask = Image.open(io.BytesIO(mask.file.read()))
    model = inpainting_ldm.__wrapped__.model
    output = model(image, mask)
    return ImageResponse(output)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
