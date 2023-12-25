import copy
import io
import os
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import requests
from PIL import Image
from typing import List, Union
from pathlib import Path
from cllm.services.utils import get_bytes_value
from cllm.utils import get_real_path
from cllm.services.nlp.api import openai_chat_model

__ALL__ = [
    "instruct_pix2pix",
    "image_cropping",
    "image_matting",
    "draw_bbox_on_image",
    "partial_image_editing",
]


HOST = "localhost"
PORT = os.environ.get("CLLM_SERVICES_PORT", 10056)


def setup(host="localhost", port=10049):
    global HOST, PORT
    HOST = host
    PORT = port


def image_cropping(image: str | Path, object: List[dict], **kwargs):
    """
    bbox format: {'score': 0.997, 'label': 'bird', 'box': {'xmin': 69, 'ymin': 171, 'xmax': 396, 'ymax': 507}}
    """
    if object in [None, b"", []]:
        return None

    if isinstance(image, (str, Path)):
        image = Image.open(get_real_path(image)).convert("RGB")
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    w, h = image.size
    cropped_images = []
    for box in object:
        box = copy.deepcopy(box["box"])
        box = unify_bbox(box, w, h)
        (left, upper, right, lower) = (
            box["xmin"],
            box["ymin"],
            box["xmax"],
            box["ymax"],
        )
        cropped_image = image.crop((left, upper, right, lower))
        # cropped_image.save('test.png')
        img_stream = io.BytesIO()
        cropped_image.save(img_stream, format="png")
        img_stream.seek(0)
        cropped_images.append(img_stream.getvalue())
    if len(cropped_images) == 0:
        return None
    return cropped_images


def image_matting(image: str | Path, mask: Union[str, bytes, List], **kwargs):
    """
    {'score': 0.999025,
    'label': 'person',
    'mask': <PIL.Image.Image image mode=L size=386x384>}
    """
    if mask in [None, b"", []]:
        return None
    image = Image.open(get_bytes_value(image)).convert("RGB")

    mask = copy.deepcopy(mask)
    if isinstance(mask, List):
        mask_list = []
        for m in mask:
            if isinstance(m, dict):
                mask_list.append(get_bytes_value(m["mask"]))
            else:
                mask_list.append(get_bytes_value(m))
        mask = combine_masks(mask_list)
    elif isinstance(mask, str):
        mask = get_bytes_value(mask)

    mask = Image.open(mask).convert("L")

    mask = np.array(mask) > 0
    image = np.array(image)
    image = image * np.expand_dims(mask, -1)
    img_stream = io.BytesIO()
    image.save(img_stream, format="png")
    img_stream.seek(0)
    return img_stream.getvalue()


def unify_bbox(bbox, w, h):
    bbox["xmin"] = (
        bbox["xmin"] if isinstance(bbox["xmin"], int) else int(bbox["xmin"] * w)
    )

    bbox["ymin"] = (
        bbox["ymin"] if isinstance(bbox["ymin"], int) else int(bbox["ymin"] * h)
    )
    bbox["xmax"] = (
        bbox["xmax"] if isinstance(bbox["xmax"], int) else int(bbox["xmax"] * w)
    )
    bbox["ymax"] = (
        bbox["ymax"] if isinstance(bbox["ymax"], int) else int(bbox["ymax"] * h)
    )
    return bbox


def draw_bbox_on_image(image: str | Path, bbox: list, **kwargs):
    if isinstance(image, (str, Path)):
        image = Image.open(get_real_path(image)).convert("RGB")
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    image = image.copy()
    w, h = image.size
    for box in bbox:
        box = copy.deepcopy(box["box"])
        box = unify_bbox(box, w, h)
        (left, upper, right, lower) = (
            box["xmin"],
            box["ymin"],
            box["xmax"],
            box["ymax"],
        )
        draw = ImageDraw.Draw(image)
        font_width = int(
            min(box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]) * 0.01
        )
        draw.rectangle(((left, upper), (right, lower)), outline="Red", width=font_width)
    img_stream = io.BytesIO()
    image.save(img_stream, format="png")
    img_stream.seek(0)
    # image = Image.save(image, format='png')
    return img_stream.getvalue()


def _imagetext2image(image, text, endpoint, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/{endpoint}"
    data = {"text": text}
    files = {"image": (image, get_bytes_value(image))}
    response = requests.post(url, files=files, data=data)
    return response.content


def instruct_pix2pix(image, text, **kwargs):
    return _imagetext2image(image, text, endpoint="instruct_pix2pix", **kwargs)


def partial_image_editing(
    image: str | bytes, mask: str | list | bytes, prompt: str, **kwargs
):
    if mask in [None, b"", []]:
        return None

    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/partial_image_editing"
    human_msg = f"""Your task is to extract the prompt from input. Here is examples:

    Input:
    Replace the masked object in the given image with a yellow horse

    Answer:
    a yellow horse

    Input:
    Use the c1s5af_mask.png in to replace the object with a man in the image

    Answer:
    a man

    Input:
    Modify the given image by replacing the object indicated in the mask with a bouquet of flowers

    Answer:
    with a bouquet of flowers

    Input:
    Use the 7a3c72_mask.png file to replace the object in the a9430b_image.png with a bus colored yellow and red with the number 5 on its front sign

    Answer:
    a bus colored yellow and red with the number 5 on its front sign.

    Input:
    Replace the masked area in image with a fat boy wearing a black jacket.

    Answer:
    a fat boy wearing a black jacket

    Input:
    {prompt}

    Answer:
    """
    extracted_prompt = openai_chat_model(human_msg)
    data = {"prompt": extracted_prompt}
    if isinstance(mask, List):
        mask_list = []
        for m in mask:
            if isinstance(m, dict):
                mask_list.append(get_bytes_value(m["mask"]))
            else:
                mask_list.append(get_bytes_value(m))
        mask = combine_masks(mask_list)

    files = {
        "image": (image, get_bytes_value(image)),
        "mask": ("mask", get_bytes_value(mask)),
    }
    response = requests.post(url, files=files, data=data)
    return response.content


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


def inpainting_ldm_general(image, mask: Union[str, bytes, List], **kwargs):
    if mask in [None, b"", []]:
        return get_bytes_value(image)

    mask = copy.deepcopy(mask)
    if isinstance(mask, List):
        mask_list = []
        for m in mask:
            if isinstance(m, dict):
                mask_list.append(get_bytes_value(m["mask"]))
            else:
                mask_list.append(get_bytes_value(m))
        mask = combine_masks(mask_list)
    elif isinstance(mask, str):
        mask = get_bytes_value(mask)
        # mask = Image.open(mask).convert("1")

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
