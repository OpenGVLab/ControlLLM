import copy
from typing import Union, List, Dict
from PIL import Image, ImageChops
import io
import os

import requests
from cllm.servcies.utils import get_bytes_value

__ALL__ = [
    "inpainting_ldm",
]


HOST = "localhost"
PORT = os.environ.get("CLLM_SERVICES_PORT", 10056)


def setup(host="localhost", port=10052):
    global HOST, PORT
    HOST = host
    PORT = port


def combine_masks(mask_images):
    if mask_images is None or len(mask_images) == 0:
        return None

    # Create a new blank image to store the combined mask
    combined_mask = Image.open(io.BytesIO(mask_images[0])).convert("1")

    # Iterate through each mask image and combine them
    for mask_image in mask_images:
        mask = Image.open(io.BytesIO(mask_image)).convert("1")
        combined_mask = ImageChops.logical_or(combined_mask, mask)
    stream = io.BytesIO()
    combined_mask.save(stream, "png")
    stream.seek(0)
    # return {"label": mask_images[0]["label"], "mask": stream.getvalue()}
    return stream.getvalue()


def inpainting_ldm_general(image, mask: Union[bytes, List], **kwargs):
    if mask in [None, b"", []]:
        return get_bytes_value(image)

    mask = copy.deepcopy(mask)
    if isinstance(mask, List):
        if not isinstance(mask[0], dict):
            mask_list = get_bytes_value(mask)
        else:
            mask_list = []
            for m in mask:
                mask_list.append(get_bytes_value(m["mask"]))
        mask = combine_masks(mask_list)

    return inpainting_ldm(image, mask, **kwargs)


def inpainting_ldm(image, mask, **kwargs):
    if mask in [None, b""]:
        return get_bytes_value(image)

    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/inpainting_ldm"
    files = {
        "image": (image, get_bytes_value(image)),
        "mask": get_bytes_value(mask),
    }
    response = requests.post(url, files=files)
    return response.content
