import argparse
import os

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Response
from PIL import Image
import io
import uuid

from .tools import Anything2Image

parser = argparse.ArgumentParser(description="Anything2Image API")
parser.add_argument("--port", type=int, default=10049, help="Port")
args = parser.parse_args()

app = FastAPI()
model = Anything2Image('cuda:0')

TMP_DIR = 'anything2image_tmp'
os.makedirs(TMP_DIR, exist_ok=True)


def get_bytes_value(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='png')
    return img_byte_arr.getvalue()


@app.post("/audio2image")
async def audio2image(audio: UploadFile = File(None)):
    image_bytes = image.file.read()
    image: Image = Image.open(io.BytesIO(image_bytes))
    image_path = os.path.join(TMP_DIR, str(uuid.uuid3))
    image.save(image_path)
    output = model.audio2image(image_path)
    buffer = get_bytes_value(output)
    return Response(content=buffer, media_type="image/jpg")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=args.port)
