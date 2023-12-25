import argparse

import uvicorn
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import io

from .tools import *

from cllm.services import app, pool

parser = argparse.ArgumentParser(description="Image Transformation API")
parser.add_argument("--host", type=str, default="localhost", help="Host")
parser.add_argument("--port", type=int, default=10049, help="Port")
args = parser.parse_args()


def ImageResponse(image):
    img_stream = io.BytesIO()
    image.save(img_stream, format="png")
    img_stream.seek(0)

    return StreamingResponse(img_stream, media_type="image/png")


@app.post("/image2canny")
@pool.register(lambda: Image2Canny())
async def image2canny(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = image2canny.__wrapped__.model
    output = model(image)
    return ImageResponse(output)


@app.post("/image2line")
@pool.register(lambda: Image2Line())
async def image2line(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = image2line.__wrapped__.model
    output = model(image)
    return ImageResponse(output)


@app.post("/image2hed")
@pool.register(lambda: Image2Hed())
async def image2hed(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = image2hed.__wrapped__.model
    output = model(image)
    return ImageResponse(output)


@app.post("/image2scribble")
@pool.register(lambda: Image2Scribble())
async def image2scribble(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = image2scribble.__wrapped__.model
    output = model(image)
    return ImageResponse(output)


@app.post("/image2pose")
@pool.register(lambda: Image2Pose())
async def image2pose(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = image2pose.__wrapped__.model
    output = model(image)
    return ImageResponse(output)


@app.post("/image2depth")
@pool.register(lambda: Image2Depth())
async def image2depth(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = image2depth.__wrapped__.model
    output = model(image)
    return ImageResponse(output)


@app.post("/image2normal")
@pool.register(lambda: Image2Normal())
async def image2normal(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = image2normal.__wrapped__.model
    output = model(image)
    return ImageResponse(output)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
