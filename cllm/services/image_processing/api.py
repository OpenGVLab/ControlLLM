import io
import os

import requests
from PIL import Image
from cllm.services.utils import get_bytes_value

__ALL__ = [
    "image2canny",
    "image2line",
    "image2hed",
    "image2scribble",
    "image2pose",
    "image2depth",
    "image2normal",
]


HOST = "localhost"
PORT = os.environ.get("CLLM_SERVICES_PORT", 10056)


def setup(host="localhost", port=10049):
    global HOST, PORT
    HOST = host
    PORT = port


def image2anything(image: Image, endpoint="image2line", **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/{endpoint}"
    files = {"image": (image, get_bytes_value(image))}
    response = requests.post(url, files=files)
    return response.content


def image2canny(image: Image, **kwargs):
    return image2anything(image, endpoint="image2canny", **kwargs)


def image2line(image: Image, **kwargs):
    return image2anything(image, endpoint="image2line", **kwargs)


def image2hed(image: Image, **kwargs):
    return image2anything(image, endpoint="image2hed", **kwargs)


def image2scribble(image: Image, **kwargs):
    return image2anything(image, endpoint="image2scribble", **kwargs)


def image2pose(image: Image, **kwargs):
    return image2anything(image, endpoint="image2pose", **kwargs)


def image2depth(image: Image, **kwargs):
    return image2anything(image, endpoint="image2depth", **kwargs)


def image2normal(image: Image, **kwargs):
    return image2anything(image, endpoint="image2normal", **kwargs)
