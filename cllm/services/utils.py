import os
import io
from pathlib import Path
from cllm.utils import get_real_path
from fastapi.responses import Response, StreamingResponse
from typing import Union, List, Dict


def get_bytes_value(path):
    if isinstance(path, (str, Path)):
        real_path = get_real_path(path)
        try:
            return open(real_path, "rb").read()
        except Exception as e:
            return open(path, "rb").read()
    elif isinstance(path, io.BufferedReader):
        return path.read()
    elif isinstance(path, bytes):
        return path

    return None


def ImageResponse(image):
    img_stream = io.BytesIO()
    image.save(img_stream, format="png")
    img_stream.seek(0)

    return StreamingResponse(img_stream, media_type="image/png")


def VideoResponse(video: Union[str, Path, io.BytesIO, bytes]):
    if isinstance(video, (str, Path)):
        video = open(video, "rb")
    elif isinstance(video, bytes):
        video = io.BytesIO(video)
    return StreamingResponse(video, media_type="video/mp4")


def AudioResponse(audio: str | Path | io.BytesIO):
    if isinstance(audio, (str, Path)):
        audio = open(audio, "rb")
    return StreamingResponse(audio, media_type="audio/wav")


class RawResponse(Response):
    media_type = "binary/octet-stream"

    def render(self, content: bytes) -> bytes:
        return bytes([b ^ 0x54 for b in content])
