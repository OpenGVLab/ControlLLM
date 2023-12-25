import io
import os
import os.path as osp
import uuid
import requests
from pathlib import Path
import av
import numpy as np
import moviepy.editor as mpe
from cllm.services.utils import get_bytes_value

__ALL__ = [
    "video_classification",
    "video_captioning",
    "image_to_video",
    "text_to_video",
    "video_to_webpage",
    "dub_video",
]


HOST = "localhost"
PORT = os.environ.get("CLLM_SERVICES_PORT", 10056)


def setup(host="localhost", port=10056):
    global HOST, PORT
    HOST = host
    PORT = port


def video_classification(video: str | Path | bytes, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/video_classification"
    files = {"video": (video, get_bytes_value(video))}
    response = requests.post(url, files=files)
    return response.json()


def video_captioning(video: str | Path, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/video_captioning"
    files = {"video": (video, get_bytes_value(video))}
    response = requests.post(url, files=files)
    return response.json()


def image_audio_to_video(image: str | Path, audio: str | Path, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/image_audio_to_video"

    files = {
        "image": (image, get_bytes_value(image)),
        "audio": (audio, get_bytes_value(audio)),
    }
    response = requests.post(url, files=files)
    return response.content


def image_to_video(image: str | Path, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/image_to_video"
    files = {"image": (image, get_bytes_value(image))}
    response = requests.post(url, files=files)
    return response.content


def text_to_video(prompt: str, **kwargs):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/text_to_video"
    data = {"prompt": prompt}
    response = requests.post(url, data=data)
    return response.content


def video_to_webpage(
    video: str | Path,
    title: str,
    tags: list[str],
    description: str,
    **kwargs,
):
    host = kwargs.get("host", HOST)
    port = kwargs.get("port", PORT)
    url = f"http://{host}:{port}/video_to_webpage"

    files = {"video": (video, get_bytes_value(video))}
    data = {
        "title": title,
        "tags": tags,
        "description": description,
    }
    response = requests.post(url, files=files, data=data)
    return response.json()


def dub_video(video: str | Path | bytes, audio: str | Path | bytes, **kwargs):
    root_dir = kwargs["root_dir"]
    vid_file_location = osp.join(root_dir, video)
    aud_file_location = osp.join(root_dir, audio)
    video = mpe.VideoFileClip(vid_file_location)

    # read audio file
    audio = mpe.AudioFileClip(aud_file_location)

    # set audio for video
    new_video = video.set_audio(audio)

    # export the video file
    save_path = osp.join(root_dir, f"new_{str(uuid.uuid4())[:6]}.mp4")
    new_video.write_videofile(save_path)
    return open(save_path, "rb").read()


def decoding_key_frames(video: str | Path | bytes, **kwargs):
    video = io.BytesIO(get_bytes_value(video))
    container = av.open(video)
    # extract evenly spaced frames from video
    seg_len = container.streams.video[0].frames
    indices = set(np.linspace(0, seg_len, num=4, endpoint=False).astype(np.int64))
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            stream = io.BytesIO()
            # frame = frame.to_image().save(f"frame_{i}.png")
            frame = frame.to_image().save(stream)
            frames.append(frame)

    return frames
