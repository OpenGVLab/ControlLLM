import os

os.environ["CURL_CA_BUNDLE"] = ""

import argparse
import codecs
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, Response

from PIL import Image
import io
import pickle
import json

from .tools import *
from cllm.services import app, pool
from cllm.services.utils import ImageResponse
from ..hf_pipeline import HuggingfacePipeline

parser = argparse.ArgumentParser(description="Image Perception API")
parser.add_argument("--host", type=str, default="localhost", help="Host")
parser.add_argument("--port", type=int, default=10049, help="Port")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
args = parser.parse_args()


def SAM():
    return SegmentAnythingStateful(args.device)


@app.post("/object_detection")
@pool.register(lambda: HuggingfacePipeline("object-detection", args.device))
async def object_detection(image: UploadFile = File(None)):
    image.file.seek(0)
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = object_detection.__wrapped__.model
    output = model(image)
    return JSONResponse(output)


@app.post("/image_classification")
@pool.register(lambda: HuggingfacePipeline("image-classification", args.device))
async def image_classification(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = image_classification.__wrapped__.model
    output = model(image)
    return JSONResponse(output)


@app.post("/image_to_text")
@pool.register(lambda: HuggingfacePipeline("image-to-text", args.device))
async def image_to_text(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = image_to_text.__wrapped__.model
    output = model(image)
    return JSONResponse(output)


@app.post("/ocr")
@pool.register(lambda: OCR(args.device))
async def ocr(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = ocr.__wrapped__.model
    output = model(image)
    return JSONResponse(output)


@app.post("/segment_objects")
@pool.register(lambda: HuggingfacePipeline("image-segmentation", args.device))
async def segment_objects(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = segment_objects.__wrapped__.model
    output = model(image)
    pickled = codecs.encode(pickle.dumps(output), "base64").decode()
    return JSONResponse({"data": pickled})


@app.post("/visual_grounding")
@pool.register(lambda: VisualGrounding(args.device))
async def visual_grounding(query: str = Form(...), image: UploadFile = File(...)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = visual_grounding.__wrapped__.model
    coordinates = model(image, query)
    print(coordinates)
    return JSONResponse(coordinates)


@app.post("/captioning_blip")
@pool.register(lambda: BLIPImageCaptioning(args.device))
async def captioning_blip(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = captioning_blip.__wrapped__.model
    output = model(image)
    return output


@app.post("/segment_all")
@pool.register(SAM)
async def segment_all(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = segment_all.__wrapped__.model
    output = model(image)
    return ImageResponse(output)


@app.post("/set_image")
@pool.register(SAM)
async def set_image(image: UploadFile = File(None)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = set_image.__wrapped__.model
    output = model.set_image(image)
    return Response(content=output)


@app.post("/segment_by_mask")
@pool.register(SAM)
async def segment_by_mask(mask: UploadFile = File(None), image_id: str = Form(...)):
    image_bytes = mask.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    model = segment_by_mask.__wrapped__.model
    output = model.segment_by_mask(image, image_id)
    return ImageResponse(output)


@app.post("/segment_by_points")
@pool.register(SAM)
async def segment_by_points(points: str | list = Body(...), image_id: str = Form(...)):
    if isinstance(points, str):
        points = json.loads(points)

    model = segment_by_points.__wrapped__.model
    output = model.segment_by_points(points, image_id)
    return ImageResponse(output)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
