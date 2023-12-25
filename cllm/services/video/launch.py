import argparse
import os
import os.path as osp
import io
from pathlib import Path
from typing import Union

import uvicorn
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from .tools import *
from cllm.services.utils import VideoResponse
from cllm.services import app, pool
from ..hf_pipeline import HuggingfacePipeline

parser = argparse.ArgumentParser(description="Video API")
parser.add_argument("--host", type=str, default="localhost", help="Host")
parser.add_argument("--port", type=int, default=10049, help="Port")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
args = parser.parse_args()


RESOURCE_ROOT = os.environ.get("SERVER_ROOT", "./server_resources")
os.makedirs(RESOURCE_ROOT, exist_ok=True)


# def VideoResponse(video: Union[str, Path, io.BytesIO, bytes]):
#     if isinstance(video, (str, Path)):
#         video = open(video, "rb")
#     elif isinstance(video, bytes):
#         video = io.BytesIO(video)
#     return StreamingResponse(video, media_type="video/mp4")


@app.post("/video_classification")
@pool.register(lambda: HuggingfacePipeline("video-classification", args.device))
async def video_classification(video: UploadFile = File(None)):
    model = video_classification.__wrapped__.model

    vid_name = osp.basename(video.filename)
    vid_name = osp.basename(video.filename)
    print(f"video_captioning --- vid_name: {vid_name}")
    vid_file_location = osp.join(RESOURCE_ROOT, vid_name)
    with open(vid_file_location, "wb+") as file_object:
        file_object.write(video.file.read())

    output = model(vid_file_location)
    os.remove(vid_file_location)

    return JSONResponse(output)


@app.post("/video_captioning")
@pool.register(lambda: TimeSformerGPT2VideoCaptioning(args.device))
async def video_captioning(video: UploadFile = File(None)):
    video.file.seek(0)
    model = video_captioning.__wrapped__.model
    vid_name = osp.basename(video.filename)
    vid_file_location = osp.join(RESOURCE_ROOT, vid_name)
    with open(vid_file_location, "wb+") as file_object:
        file_object.write(video.file.read())

    output = model(vid_file_location)
    print(f"video_captioning output: {output}")
    os.remove(vid_file_location)

    return JSONResponse(output)


@app.post("/image_to_video")
@pool.register(lambda: Image2Video(args.device))
async def image_to_video(image: UploadFile = File(None)):
    model = image_to_video.__wrapped__.model
    image = Image.open(image.file).convert("RGB")

    output = model(image)
    return VideoResponse(output)


@app.post("/text_to_video")
@pool.register(lambda: Text2Video(args.device))
async def text_to_video(prompt: str = Form(...)):
    model = text_to_video.__wrapped__.model
    output = model(prompt)
    return VideoResponse(output)


@app.post("/image_audio_to_video")
@pool.register(lambda: ImageAudio2Video(args.device))
async def image_audio_to_video(
    image: UploadFile = File(None), audio: UploadFile = File(None)
):
    model = image_audio_to_video.__wrapped__.model
    img_name = osp.basename(image.filename)
    img_file_location = osp.join(RESOURCE_ROOT, img_name)
    aud_name = osp.basename(audio.filename)
    aud_file_location = osp.join(RESOURCE_ROOT, aud_name)
    with open(img_file_location, "wb+") as file_object:
        file_object.write(image.file.read())
    with open(aud_file_location, "wb+") as file_object:
        file_object.write(audio.file.read())

    output = model(img_file_location, aud_file_location)
    os.remove(img_file_location)
    os.remove(aud_file_location)
    return VideoResponse(output)


@app.post("/video_to_webpage")
@pool.register(lambda: Video2WebPage(args.device))
async def video_to_webpage(
    video: UploadFile = File(None),
    title: str = Form(...),
    tags: set[str] = Form(...),
    description: str = Form(...),
):
    model = video_to_webpage.__wrapped__.model
    vid_name = osp.basename(video.filename)
    html_str = model(vid_name, title, tags, description)
    return JSONResponse(html_str)


@app.post("/dub_video")
@pool.register(lambda: DubVideo(args.device))
async def dub_video(video: UploadFile = File(None), audio: UploadFile = File(None)):
    model = dub_video.__wrapped__.model
    vid_name = osp.basename(video.filename)
    vid_file_location = osp.join(RESOURCE_ROOT, vid_name)
    with open(vid_file_location, "wb+") as file_object:
        file_object.write(video.file.read())

    aud_name = osp.basename(audio.filename)
    aud_file_location = osp.join(RESOURCE_ROOT, aud_name)

    with open(aud_file_location, "wb+") as file_object:
        file_object.write(audio.file.read())

    new_video_file = model(vid_file_location, aud_file_location)
    return VideoResponse(new_video_file)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
