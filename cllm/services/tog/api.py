import os
import requests

__ALL__ = ["tog", "task_decomposer"]


HOST = "localhost"
PORT = os.environ.get("TOG_SERVICE_PORT", 10052)


def setup(host="localhost", port=10052):
    global HOST, PORT
    HOST = host
    PORT = port


def tog(request, subtasks, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    stream = kwargs.get("stream", False)
    url = f"http://{host}:{port}/tog"
    data = {"request": request, "subtasks": subtasks, "stream": stream}
    response = requests.post(url, data=data, stream=stream)
    # if not stream:
    #     response = response.content.decode("utf-8")
    # print(f"response.json(): {response.json()}")
    return response.json()


def task_decomposer(request, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    stream = kwargs.get("stream", False)
    url = f"http://{host}:{port}/task_decomposer"
    data = {"request": request, "stream": stream}
    response = requests.post(url, data=data, stream=stream)
    # if not stream:
    #     response = response.content.decode("utf-8")
    # return response.content.decode("utf-8")
    return response.json()
