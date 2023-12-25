import os
import os.path as osp
from pathlib import Path
import json
from .base import DataType
from cllm.utils import get_real_path

FILE_EXT = {
    "image": ["png", "jpeg", "jpg", "gif", "bmp", "tiff", "webp"],
    "video": ["mp4", "mov", "avi", "mkv"],
    "audio": ["wav", "mp3"],
}


class Container:
    def __init__(self, name, rtype, value) -> None:
        self.name = name
        self.rtype = rtype
        self.value = value

    def to_chatbot(self):
        pass

    def __str__(self):
        pass

    def __repr__(self) -> str:
        return str(self)


class File(Container):
    def to_chatbot(self):
        return str(self.value)

    @property
    def filename(self):
        return os.path.basename(self.value)

    def __str__(self):
        return f"`{self.filename}`"


class HTML(File):
    def to_chatbot(self):
        return str(self.value)

    def __str__(self):
        return f"`{self.filename}`"


class Image(File):
    def __str__(self):
        return f"`{self.filename}`"


class Video(File):
    def __str__(self):
        return f"`{self.filename}`"


class Audio(File):
    def __str__(self):
        return f"`{self.filename}`"


class Text(Container):
    def to_chatbot(self):
        if isinstance(self.value, str):
            return self.value
        elif isinstance(self.value, (list, tuple, dict)):
            return json.dumps(self.value, indent=2)
        return self.value

    def __str__(self):
        if isinstance(self.value, (list, dict)):
            return json.dumps(self.value)
        elif isinstance(self.value, str):
            return self.value
        return str(self.value)


def auto_type(name, rtype, value):
    if value is None:
        return None
    if "image" in str(rtype):
        return Image(name, rtype, get_real_path(value))
    if DataType.VIDEO == rtype:
        return Video(name, rtype, get_real_path(value))
    if DataType.AUDIO == rtype:
        return Audio(name, rtype, get_real_path(value))
    if DataType.HTML == rtype:
        return HTML(name, rtype, get_real_path(value))
    return Text(name, rtype, value)
