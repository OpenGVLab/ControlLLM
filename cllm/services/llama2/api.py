import os
import requests

__ALL__ = ["llama2_chat"]


HOST = "localhost"
PORT = os.environ.get("CLLM_LLAMA2_PORT", 10051)


def setup(host="localhost", port=10051):
    global HOST, PORT
    HOST = host
    PORT = port


def llama2_chat(messages, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/llama2_chat"
    response = requests.post(url, json=messages)
    return response.content.decode()
