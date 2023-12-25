import os
import requests

PORT = os.environ.get("CLLM_SERVICES_PORT", 10056)


def audio2image(audio):
    url = "http://localhost:10049/chat"
    # files = {"image": open("assets/ADE_val_00000529.jpg", "rb")}
    data = {"audio": audio}
    response = requests.post(url, data=data)
    return response.json()
