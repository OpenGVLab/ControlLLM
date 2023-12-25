import io
import os

import requests
from PIL import Image
from cllm.services.utils import get_bytes_value


__ALL__ = [
    "text2image",
    "cannytext2image",
    "linetext2image",
    "hedtext2image",
    "scribbletext2image",
    "posetext2image",
    "segtext2image",
    "depthtext2image",
    "normaltext2image" "image2image",
]


HOST = "localhost"
PORT = os.environ.get("CLLM_SERVICES_PORT", 10056)


def setup(host="localhost", port=10049):
    global HOST, PORT
    HOST = host
    PORT = port


def text2image(text, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/text2image"
    data = {"text": text}
    response = requests.post(url, data=data)
    return response.content


def image2image(image, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/image2image"
    files = {"image": (image, get_bytes_value(image))}
    response = requests.post(url, files=files)
    return response.content


def _imagetext2image(image, text, endpoint, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/{endpoint}"
    data = {"text": text}
    files = {"image": (image, get_bytes_value(image))}
    response = requests.post(url, files=files, data=data)
    # image = Image.open(io.BytesIO(response.content))
    # image = io.BytesIO(response.content)
    # return image
    return response.content


def cannytext2image(edge, text, **kwargs):
    return _imagetext2image(edge, text, endpoint="cannytext2image", **kwargs)


def linetext2image(line, text, **kwargs):
    return _imagetext2image(line, text, endpoint="linetext2image", **kwargs)


def hedtext2image(hed, text, **kwargs):
    return _imagetext2image(hed, text, endpoint="hedtext2image", **kwargs)


def scribbletext2image(scribble, text, **kwargs):
    return _imagetext2image(scribble, text, endpoint="scribbletext2image", **kwargs)


def posetext2image(pose, text, **kwargs):
    return _imagetext2image(pose, text, endpoint="posetext2image", **kwargs)


def segtext2image(segmentation, text, **kwargs):
    return _imagetext2image(segmentation, text, endpoint="segtext2image", **kwargs)


def depthtext2image(depth, text, **kwargs):
    return _imagetext2image(depth, text, endpoint="depthtext2image", **kwargs)


def normaltext2image(normal, text, **kwargs):
    return _imagetext2image(normal, text, endpoint="normaltext2image", **kwargs)
