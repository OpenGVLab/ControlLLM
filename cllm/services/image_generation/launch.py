import argparse

import uvicorn
from fastapi import UploadFile, File, Form
from PIL import Image
import io

from .tools import *
from cllm.services import app, pool
from cllm.services.utils import ImageResponse


parser = argparse.ArgumentParser(description="Image Generation API")
parser.add_argument("--host", type=str, default="localhost", help="Host")
parser.add_argument("--port", type=int, default=10049, help="Port")
parser.add_argument("--device", type=str, default="cuda:0", help="Port")
args = parser.parse_args()


# def ImageResponse(image):
#     img_stream = io.BytesIO()
#     image.save(img_stream, format="png")
#     img_stream.seek(0)

#     return StreamingResponse(img_stream, media_type="image/png")


# @app.post("/text2image")
# @pool.register(lambda: Text2Image(args.device))
# async def text2image(text: str = Form(...)):
#     model = text2image.__wrapped__.model
#     output = model(text)
#     return ImageResponse(output)


@app.post("/text2image")
@pool.register(lambda: PixArtAlpha(args.device))
async def text2image(text: str = Form(...)):
    model = text2image.__wrapped__.model
    output = model(text)
    return ImageResponse(output)


@app.post("/image2image")
@pool.register(lambda: Image2Image(args.device))
async def image2image(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = image2image.__wrapped__.model
    output = model(image)
    return ImageResponse(output)


@app.post("/cannytext2image")
@pool.register(lambda: CannyText2Image(args.device))
async def cannytext2image(image: UploadFile = File(None), text: str = Form(...)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = cannytext2image.__wrapped__.model
    output = model(image, text)
    return ImageResponse(output)


@app.post("/linetext2image")
@pool.register(lambda: LineText2Image(args.device))
async def linetext2image(image: UploadFile = File(None), text: str = Form(...)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = linetext2image.__wrapped__.model
    output = model(image, text)
    return ImageResponse(output)


@app.post("/hedtext2image")
@pool.register(lambda: HedText2Image(args.device))
async def hedtext2image(image: UploadFile = File(None), text: str = Form(...)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = hedtext2image.__wrapped__.model
    output = model(image, text)
    return ImageResponse(output)


@app.post("/scribbletext2image")
@pool.register(lambda: ScribbleText2Image(args.device))
async def scribbletext2image(image: UploadFile = File(None), text: str = Form(...)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = scribbletext2image.__wrapped__.model
    output = model(image, text)
    return ImageResponse(output)


@app.post("/posetext2image")
@pool.register(lambda: PoseText2Image(args.device))
async def posetext2image(image: UploadFile = File(None), text: str = Form(...)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = posetext2image.__wrapped__.model
    output = model(image, text)
    return ImageResponse(output)


@app.post("/segtext2image")
@pool.register(lambda: SegText2Image(args.device))
async def segtext2image(image: UploadFile = File(None), text: str = Form(...)):
    image.file.seek(0)
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = segtext2image.__wrapped__.model
    output = model(image, text)
    return ImageResponse(output)


@app.post("/depthtext2image")
@pool.register(lambda: SegText2Image(args.device))
async def depthtext2image(image: UploadFile = File(None), text: str = Form(...)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = depthtext2image.__wrapped__.model
    output = model(image, text)
    return ImageResponse(output)


@app.post("/normaltext2image")
@pool.register(lambda: SegText2Image(args.device))
async def normaltext2image(image: UploadFile = File(None), text: str = Form(...)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = normaltext2image.__wrapped__.model
    output = model(image, text)
    return ImageResponse(output)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
