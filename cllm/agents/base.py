from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List
import json
from pathlib import Path
from collections import OrderedDict


@dataclass
class Action:
    """The action represent an assignment.
        `output = tool_name(**inputs)`

    Examples:
        >>> mask = segmentation_by_mask(image=image, prompt_mask=prompt_mask)
        >>> image = image_inpainting(image=image, mask=mask)
    """

    tool_name: str = (None,)
    inputs: dict = (None,)
    outputs: List[str] = (None,)

    def __str__(self) -> str:
        args = ", ".join([f"{k}={v}" for k, v in self.inputs.items()])
        return "{} = {}(".format(", ".join(self.outputs), self.tool_name) + args + ")"

    def dict(self):
        args = {str(k): str(v) for k, v in self.inputs.items()}
        # args = {str(item["name"]): str(item["value"]) for item in self.inputs}
        rets = [o if isinstance(o, str) else str(o) for o in self.outputs]
        return {
            "tool": self.tool_name,
            "inputs": args,
            "outputs": rets,
        }


class DataType(Enum):
    TEXT = "text"
    TAGS = "tags"
    TITLE = "title"
    # HTML = "text.html"
    HTML = "html"
    LOCATION = "location"
    WEATHER = "weather"
    TIME = "time"

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    ANY = "any"
    NONE = "none"

    SEGMENTATION = "image.segmentation"
    EDGE = "image.edge"
    LINE = "image.line"
    HED = "image.hed"
    CANNY = "image.canny"
    SCRIBBLE = "image.scribble"
    POSE = "image.pose"
    DEPTH = "image.depth"
    NORMAL = "image.normal"

    MASK = "image.mask"  # SAM mask
    POINT = "point"
    BBOX = "bbox"  # {'label': 'dog', 'box': [1,2,3,4], 'score': 0.9}
    CATEGORY = "category"

    LIST = "list"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, self.__class__):
            return self.value == other.value
        else:
            return False


@dataclass
class Resource:
    name: str
    type: DataType
    value: None
    # description: str = None

    def dict(self):
        return {
            "name": self.name,
            "type": str(self.type),
            "value": str(self.value),
            # "description": self.description,
        }


@dataclass
class Tool:
    class Domain(Enum):
        IMAGE_PERCEPTION = "image-perception"
        IMAGE_GENERATION = "image-generation"
        IMAGE_EDITING = "image-editing"
        IMAGE_PROCESSING = "image-processing"
        AUDIO_PERCEPTION = "audio-perception"
        AUDIO_GENERATION = "audio-generation"
        VIDEO_PERCEPTION = "video-perception"
        VIDEO_GENERATION = "video-generation"
        VIDEO_PROCESSING = "video-processing"
        VIDEO_EDITING = "video-editing"
        VIDEO_CUTTING = "video-cutting"
        NATURAL_LANGUAGE_PROCESSING = "natural-language-processing"
        CODE_GENERATION = "code-generation"
        VISUAL_QUESTION_ANSWERING = "visual-question-answering"
        QUESTION_ANSWERING = "question-answering"
        GENERAL = "general"

        def __str__(self):
            return self.value

    @dataclass
    class Argument:
        name: str
        type: DataType
        description: str

        def dict(self):
            return {
                "name": self.name,
                "type": str(self.type),
                "description": self.description,
            }

    name: str
    description: str
    domain: Domain
    model: Callable

    usages: List[str] = field(default_factory=lambda: [])
    args: List[Argument] = field(default_factory=lambda: [])
    returns: List[Argument] = field(default_factory=lambda: [])

    def dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "domain": str(self.domain),
            "args": [a.dict() for a in self.args],
            "returns": [r.dict() for r in self.returns],
        }


NON_FILE_TYPES = [
    DataType.TAGS,
    DataType.TEXT,
    DataType.TITLE,
    DataType.BBOX,
    DataType.CATEGORY,
    DataType.LIST,
    DataType.LOCATION,
    DataType.POINT,
    DataType.WEATHER,
    DataType.TIME,
]


if __name__ == "__main__":
    s = [
        [Action("a", {"aa": [Path("/a/d/e/t.txt")]}, [Path("/a/aa.txt")])],
        Action("b", {"bb": "bbb"}, ["bbb"]),
    ]
    print(json.dumps(s, indent=4, default=lambda o: o.dict()))
