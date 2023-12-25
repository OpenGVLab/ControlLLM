import io
import os
import uuid
import requests

from cllm.services.nlp.api import openai_chat_model
from cllm.services.utils import get_bytes_value

__ALL__ = [
    "audio_classification",
    "automatic_speech_recognition",
    "text_to_speech",
]


HOST = "localhost"
PORT = os.environ.get("CLLM_SERVICES_PORT", 10056)


def setup(host="localhost", port=10057):
    global HOST, PORT
    HOST = host
    PORT = port


def audio_classification(audio, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/audio_classification"
    if isinstance(audio, str):
        audio = open(audio, "rb").read()
    files = {"audio": (audio, get_bytes_value(audio))}
    response = requests.post(url, files=files)
    return response.json()


def automatic_speech_recognition(audio: str, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/automatic_speech_recognition"
    # audio_file = open(audio, "rb")
    files = {"audio": (audio, get_bytes_value(audio))}
    response = requests.post(url, files=files)
    return response.json()


def text_to_speech(text: str, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    human_msg = f"""Your task is to extract the prompt from input. Here is examples:

    Input:
    translate the text into speech: \"Hope is the thing with feathers That perches in the soul, And sings the tune without the words, And never stops at all\"

    Answer:
    Hope is the thing with feathers That perches in the soul, And sings the tune without the words, And never stops at all

    Input:
    Can you help me transcribe the text into audio: I have a dream that one day this nation will rise up and live out the true meaning of its creed: We hold these truths to be self-evident, that all men are created equal.I have a dream that one day on the red hills of Georgia, the sons of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood. I have a dream that one day even the state of Mississippi, a state sweltering with the heat of injustice, sweltering with the heat of oppression, will be transformed into an oasis of freedom and justice. I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character.

    Answer:
    I have a dream that one day this nation will rise up and live out the true meaning of its creed: We hold these truths to be self-evident, that all men are created equal.I have a dream that one day on the red hills of Georgia, the sons of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood. I have a dream that one day even the state of Mississippi, a state sweltering with the heat of injustice, sweltering with the heat of oppression, will be transformed into an oasis of freedom and justice. I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character.

    Input:
    Create speech using the text: And so, my fellow Americans: ask not what your country can do for you — ask what you can do for your country.

    Answer:
    And so, my fellow Americans: ask not what your country can do for you — ask what you can do for your country.

    Input:
    The image features a large brown and white dog standing on a tree stump, accompanied by a small cat. The dog is positioned on the right side of the stump, while the cat is on the left side. Both animals appear to be looking at the camera, creating a captivating scene.\n\nThe dog and cat are the main focus of the image, with the dog being larger and more prominent, while the cat is smaller and positioned closer to the ground. The tree stump serves as a natural and interesting backdrop for the two animals, making the scene unique and engaging.

    Answer:
    The image features a large brown and white dog standing on a tree stump, accompanied by a small cat. The dog is positioned on the right side of the stump, while the cat is on the left side. Both animals appear to be looking at the camera, creating a captivating scene.\n\nThe dog and cat are the main focus of the image, with the dog being larger and more prominent, while the cat is smaller and positioned closer to the ground. The tree stump serves as a natural and interesting backdrop for the two animals, making the scene unique and engaging.

    Input:
    Life, thin and light-off time and time again\nFrivolous tireless\nI heard the echo, from the valleys and the heart\nOpen to the lonely soul of sickle harvesting\nRepeat outrightly, but also repeat the well-being of eventually swaying in the desert oasis\nI believe I am\nBorn as the bright summer flowers\nDo not withered undefeated fiery demon rule\nHeart rate and breathing to bear the load of the cumbersome Bored\nI heard the music, from the moon and carcass\nAuxiliary extreme aestheticism bait to capture misty\nFilling the intense life, but also filling the pure\nThere are always memories throughout the earth

    Answer:
    Life, thin and light-off time and time again\nFrivolous tireless\nI heard the echo, from the valleys and the heart\nOpen to the lonely soul of sickle harvesting\nRepeat outrightly, but also repeat the well-being of eventually swaying in the desert oasis\nI believe I am\nBorn as the bright summer flowers\nDo not withered undefeated fiery demon rule\nHeart rate and breathing to bear the load of the cumbersome Bored\nI heard the music, from the moon and carcass\nAuxiliary extreme aestheticism bait to capture misty\nFilling the intense life, but also filling the pure\nThere are always memories throughout the earth

    Input:
    {text}

    Answer:
    """
    extracted_prompt = openai_chat_model(human_msg)
    print(f"extracted_prompt: {extracted_prompt}")
    url = f"http://{host}:{port}/text_to_speech"
    data = {"text": extracted_prompt}
    response = requests.post(url, data=data)
    return response.content


def text_to_music(text: str, **kwargs):
    # print('a' * 40)
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    human_msg = f"""Your task is to extract the prompt from input. Here is examples:

    Input:
    Please generate a piece of music based on given prompt. Here is the prompt: An 80s driving pop song with heavy drums

    Answer:
    An 80s driving pop song with heavy drums

    Input:
    I would like you to provide me with a new song that represents an energetic and lively 80s pop track with prominent drums and synthesizer pads

    Answer:
    an energetic and lively 80s pop track with prominent drums and synthesizer pads

    Input:
    I'm looking for a song that has a driving pop vibe from the 80s, with heavy drums and synth pads playing in the background

    Answer:
    a driving pop vibe from the 80s, with heavy drums and synth pads playing in the background

    Input:
    Can you make a song that has a lively and energetic rhythm with prominent drums and electronic keyboard sounds in the background

    Answer:
    a lively and energetic rhythm with prominent drums and electronic keyboard sounds in the background

    Input:
    Can you make a piece of light and relaxing music

    Answer:
    a piece of light and relaxing music

    Input:
    {text}

    Answer:
    """
    extracted_prompt = openai_chat_model(human_msg)
    url = f"http://{host}:{port}/text_to_music"
    data = {"text": extracted_prompt}
    response = requests.post(url, data=data)
    return response.content
