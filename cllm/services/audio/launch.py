import argparse
import os
import uuid
import numpy as np
import os.path as osp
import uvicorn
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

from .tools import *

from cllm.services import app, pool
from cllm.services.utils import AudioResponse
from ..hf_pipeline import HuggingfacePipeline


parser = argparse.ArgumentParser(description="Audio API")
parser.add_argument("--host", type=str, default="localhost", help="Host")
parser.add_argument("--port", type=int, default=10049, help="Port")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
args = parser.parse_args()

RESOURCE_ROOT = os.environ.get("SERVER_ROOT", "./server_resources")
os.makedirs(RESOURCE_ROOT, exist_ok=True)


@app.post("/audio_classification")
@pool.register(lambda: HuggingfacePipeline("audio-classification", args.device))
async def audio_classification(audio: UploadFile = File(None)):
    bytes = audio.file.read()
    model = audio_classification.__wrapped__.model
    output = model(bytes)
    return JSONResponse(output)


@app.post("/automatic_speech_recognition")
@pool.register(lambda: HuggingfacePipeline("automatic-speech-recognition", args.device))
async def automatic_speech_recognition(audio: UploadFile = File(None)):
    bytes = audio.file.read()
    model = automatic_speech_recognition.__wrapped__.model
    output = model(bytes)
    return JSONResponse(output)


@app.post("/text_to_music")
@pool.register(lambda: Text2Music(args.device))
async def text_to_music(text: str = Form(...)):
    model = text_to_music.__wrapped__.model
    output = model(text)
    return AudioResponse(output)


@app.post("/text_to_speech")
@pool.register(
    lambda: HuggingfacePipeline("text-to-speech", args.device, model="suno/bark")
)
async def text_to_speech(text: str = Form(...)):
    model = text_to_speech.__wrapped__.model
    speech = model(text)
    save_path = osp.join(RESOURCE_ROOT, f"{str(uuid.uuid4())[:6]}_audio.wav")
    scipy.io.wavfile.write(
        save_path,
        rate=speech["sampling_rate"],
        data=speech["audio"][0].astype(np.float32),
    )
    return AudioResponse(save_path)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
