import codecs
import io
import os
import pickle
from pathlib import Path
from PIL import Image
import requests
from cllm.services.utils import get_bytes_value
from cllm.services.nlp.api import openai_chat_model

__ALL__ = [
    "object_detection",
    "image_classification",
    "ocr",
    "image_to_text",
    "segment_objects",
]


HOST = "localhost"
PORT = os.environ.get("CLLM_SERVICES_PORT", 10056)


def setup(host="localhost", port=10049):
    global HOST, PORT
    HOST = host
    PORT = port


def object_detection(image, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/object_detection"
    files = {"image": (image, get_bytes_value(image))}
    response = requests.post(url, files=files)
    return response.json()


def image_classification(image, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/image_classification"
    files = {"image": (image, get_bytes_value(image))}
    response = requests.post(url, files=files)
    return response.json()


def image_to_text(image, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/image_to_text"
    files = {"image": (image, get_bytes_value(image))}
    response = requests.post(url, files=files)
    return response.json()


def ocr(image, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/ocr"
    files = {"image": (image, get_bytes_value(image))}
    response = requests.post(url, files=files)
    return response.json()


def segment_objects(image, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/segment_objects"
    files = {"image": (image, get_bytes_value(image))}
    response = requests.post(url, files=files)
    pickled = response.json()["data"]
    output = pickle.loads(codecs.decode(pickled.encode(), "base64"))
    for o in output:
        stream = io.BytesIO()
        o["mask"].save(stream, format="png")
        stream.seek(0)
        o["mask"] = stream.getvalue()

    return output


def visual_grounding(image, query, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = rf"http://{host}:{port}/visual_grounding"
    human_msg = f"""Your task is to extract the prompt from input. Here is examples:

    Input:
    find the regin of interest in the da9619_image.png: \"An elephant in right corner\"

    Answer:
    An elephant in right corner

    Input:
    locate \"A maintenance vehicle on a railway\" in the image

    Answer:
    A maintenance vehicle on a railway

    Input:
    use visual grounding method to detect the regin of interest in the 1ba6e2_image.png: The motorcycle with the rainbow flag"

    Answer:
    The motorcycle with the rainbow flag

    Input:
    for given image, find A little baby girl with brunette hair, a pink and white dress, and is being fed frosting from her mom."

    Answer:
    A little baby girl with brunette hair, a pink and white dress, and is being fed frosting from her mom

    Input:
    find the policeman on the motorcycle in the 851522_image.png"

    Answer:
    the policeman on the motorcycle

    Input:
    The legs of a zebra shown under the neck of another zebra.

    Answer:
    The legs of a zebra shown under the neck of another zebra.

    Input:
    {query}

    Answer:
    """

    extracted_prompt = openai_chat_model(human_msg)
    files = {"image": get_bytes_value(image)}
    data = {"query": extracted_prompt}
    # image = Image.open(io.BytesIO(image)).convert("RGB")
    response = requests.post(url, data=data, files=files)

    return response.json()


def image_captioning(image, endpoint="llava", **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/{endpoint}"
    data = None
    if endpoint == "llava":
        data = {"text": "Please describe the image in details."}
    files = {"image": (image, get_bytes_value(image))}
    response = requests.post(url, files=files, data=data)
    return response.content.decode("utf-8")


def segment_all(image: str | Path, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/segment_all"
    files = {"image": (image, get_bytes_value(image))}
    response = requests.post(url, files=files)
    return response.content


def set_image(image: str | Path, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/set_image"
    files = {"image": (image, get_bytes_value(image))}
    response = requests.post(url, files=files)
    return response.content.decode()


def segment_by_mask(mask: str | Path, image_id: str, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/segment_by_mask"
    data = {"image_id": image_id}
    files = {"mask": (mask, get_bytes_value(mask))}
    response = requests.post(url, files=files, data=data)
    return response.content


def segment_by_points(points: list | tuple | str, image_id: str, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/segment_by_points"
    data = {"points": points, "image_id": image_id}
    response = requests.post(url, data=data)
    return response.content


def seg_by_mask(image, prompt_mask, **kwargs):
    image_id = set_image(image)
    mask = segment_by_mask(mask=prompt_mask, image_id=image_id)
    return mask


def seg_by_points(image, prompt_points, **kwargs):
    image_id = set_image(image)
    mask = segment_by_points(points=prompt_points, image_id=image_id)
    return mask
