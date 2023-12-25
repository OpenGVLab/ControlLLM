import io
import os
from pathlib import Path
import requests
from PIL import Image
from cllm.services.utils import get_bytes_value

__ALL__ = ["vqa_blip"]


HOST = "localhost"
PORT = os.environ.get("CLLM_SERVICES_PORT", 10056)


def setup(host="localhost", port=10049):
    global HOST, PORT
    HOST = host
    PORT = port


def image_qa(image, text, endpoint="llava", **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/{endpoint}"
    files = {"image": (image, get_bytes_value(image))}
    data = {"text": text}
    response = requests.post(url, files=files, data=data)
    return response.json()
