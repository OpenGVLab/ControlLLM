import os

from matplotlib import backend_bases
from cllm.agents.tog import Planner
import openai

from multiprocessing import set_start_method

openai.api_base = os.environ.get("OPENAI_API_BASE", None)


def test_got():
    user_request = "Generate a new image with a similar composition as b3e5f8_image.png, but with a different color scheme"
    planner = Planner(backend="local")
    subtasks, plan = planner.plan(
        user_request, {"video.mp4": "video", "audio_123.wav": "audio"}
    )

    print("User's request: ")
    print(user_request)
    print("Task decomposition: ")
    print(subtasks)
    print("Solution: ")
    print(plan)


def test_tog_api():
    from cllm.services.tog.api import tog, task_decomposer

    user_request = "Generate a new image with a similar composition as b3e5f8_image.png, but with a different color scheme"
    subtasks = task_decomposer(user_request)
    solution = tog(user_request, subtasks)
    print(solution)


# test_got_api()
if __name__ == "__main__":
    test_got()
    # test_tog_api()
