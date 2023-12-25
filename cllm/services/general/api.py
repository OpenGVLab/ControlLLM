from re import I
from typing import List
from pathlib import Path
import os
import requests

__ALL__ = ["remote_logging", "select", "count"]

HOST = "localhost"
PORT = os.environ.get("CLLM_SERVICES_PORT", 10056)


def setup(host="localhost", port=10056):
    global HOST, PORT
    HOST = host
    PORT = port


def select(**kwargs):
    if "bbox_list" in kwargs:
        list = kwargs["bbox_list"]
        condition = kwargs["condition"]
        return [l for l in list if l["label"] == condition]
    if "mask_list" in kwargs:
        list = kwargs["mask_list"]
        condition = kwargs["condition"]
        # return combine_masks([l for l in list if l['label'] == condition])
        return [l for l in list if l["label"] == condition]
    if "category_list" in kwargs:
        list = kwargs["category_list"]
        condition = kwargs["condition"]
        # return combine_masks([l for l in list if l['label'] == condition])
        return [l for l in list if l["label"] == condition]


def count(**kwargs):
    len_of_list = 0
    if "bbox_list" in kwargs:
        len_of_list = len(kwargs["bbox_list"])
    elif "mask_list" in kwargs:
        len_of_list = len(kwargs["mask_list"])

    return f"The length of the given list is {len_of_list}"


def remote_logging(
    history_msgs: list,
    task_decomposition: list,
    solution: list,
    record: str,
    like: bool,
    **kwargs,
):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/remote_logging"
    data = {
        "history_msgs": history_msgs,
        "task_decomposition": task_decomposition,
        "solution": solution,
        "record": record,
        "like": like,
    }
    response = requests.post(url, data=data)
    return response.content
