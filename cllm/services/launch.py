import argparse
import uvicorn

from cllm.services import app
from .nlp.launch import *
from .video.launch import *
from .audio.launch import *
from .image_editing.launch import *
from .image_generation.launch import *
from .image_perception.launch import *
from .image_processing.launch import *
from .vqa.launch import *
from .general.launch import *

RESOURCE_ROOT = os.environ.get("SERVER_ROOT", "./server_resources")
os.makedirs(RESOURCE_ROOT, exist_ok=True)

parser = argparse.ArgumentParser(description="TOG Services")
parser.add_argument("--host", type=str, default="localhost", help="Host")
parser.add_argument("--port", type=int, default=10056, help="Port")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
args = parser.parse_args()


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
