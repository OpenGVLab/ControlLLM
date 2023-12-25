import io
import os
import time

import requests
import json
from .llms.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

__ALL__ = [
    "text_to_text_generation",
    "title_generation",
    "text_to_tags",
    "question_answering",
    "summarization",
]


HOST = "localhost"
PORT = os.environ.get("CLLM_SERVICES_PORT", 10056)


def setup(host="localhost", port=10056):
    global HOST, PORT
    HOST = host
    PORT = port


def text_to_text_generation(text: str, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/text_to_text_generation"
    data = {"text": text}
    response = requests.post(url, data=data)
    return response.json()


def question_answering_with_context(context: str, question: str, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/question_answering_with_context"
    data = {"context": context, "question": question}
    response = requests.post(url, data=data)
    return response.json()


def openai_chat_model(input_msg: str, **kwargs):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
    chat_log = []
    default_sys_msg = "Your name is ControlLLM, an AI-powered assistant developed by OpenGVLab from Shanghai AI Lab. You need to respond to user requests based on the following information."
    sys_msg = kwargs.get("sys_msg", default_sys_msg)
    if sys_msg is not None:
        chat_log.append(SystemMessage(content=sys_msg))
    # history_msgs: list[str]
    history_msgs = []
    if "history_msgs" in kwargs:
        history_msgs = kwargs.get("history_msgs", [])

    for item in history_msgs:
        if isinstance(item[0], (list, tuple)):
            item[0] = "Received file: " + item[0][0]
        if isinstance(item[1], (list, tuple)):
            item[1] = "Generated file: " + item[1][0]
        if item[0] is not None:
            chat_log.append(HumanMessage(content=item[0]))
        if item[1] is not None:
            chat_log.append(AIMessage(content=item[1]))
        # chat_log.extend([HumanMessage(content=item[0]), AIMessage(content=item[1])])
    if not isinstance(input_msg, str):
        input_msg = json.dumps(input_msg, ensure_ascii=False)
    output = chat(chat_log + [HumanMessage(content=input_msg)])
    return output


def title_generation(text: str, **kwargs):
    question = "summarize"
    response = question_answering_with_context(text, question)
    return response


def summarization(text: str, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/summarization"
    data = {"text": text}
    response = requests.post(url, data=data)
    return response.json()


def text_to_tags(text: str, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/text_to_tags"
    data = {"text": text}
    response = requests.post(url, data=data)
    return response.json()


def get_time(location: str = None, **kwargs):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_weather(location: str | list, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/get_weather"
    if isinstance(location, list):
        t = {"CITY": "", "COUNTRY": ""}
        for l in location:
            if l["entity_group"] not in t.keys():
                continue
            if t[l["entity_group"]] == "":
                t[l["entity_group"]] = l["word"].title()
        location = ",".join([t["CITY"], t["COUNTRY"]])

    data = {"location": location}
    response = requests.post(url, data=data)
    return response.json()


def summarize_weather_condition(weather: str | list, **kwargs):
    if isinstance(weather, list):
        weather = json.dumps(weather, ensure_ascii=False)
    result = openai_chat_model(
        f"Please Summarize weather condition and make user better understand it: \n {weather}"
    )
    return result


def extract_location(text: str, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/extract_location"
    data = {"text": text}
    response = requests.post(url, data=data)
    return response.json()


def sentiment_analysis(text: str, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/sentiment_analysis"
    data = {"text": text}
    response = requests.post(url, data=data)
    return response.json()
