import argparse
from functools import partial
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse

from cllm.services.pool import ModelPool
from langchain.chat_models import ChatAnthropic, ChatGooglePalm
from cllm.services.nlp.llms.chat_models import ChatOpenAI, ChatLLAMA2, MessageMemory

from . import TaskSolver, TaskDecomposer
from .configs.tog_config import config

# from .llm.llama2 import ChatLLAMA2

parser = argparse.ArgumentParser(description="Thoughts-on-Graph API")
parser.add_argument("--host", type=str, default="localhost", help="Host")
parser.add_argument("--port", type=int, default=10052, help="Port")
parser.add_argument("--llm", type=str, default="openai", help="Backend LLM")
parser.add_argument("--device", type=str, default="cuda:0", help="Port")
args = parser.parse_args()

app = FastAPI()
pool = ModelPool()


MODELS = {
    "openai": ChatOpenAI,
    "claude": ChatAnthropic,
    "google": ChatGooglePalm,
    "llama2": ChatLLAMA2,
    "gpt4": partial(ChatOpenAI, model_name="gpt-4"),
}


class TaskSolverWrapper:
    def __init__(self, device) -> None:
        cfg = config
        llm = MODELS[args.llm](
            temperature=0.1,
        )
        self.got = TaskSolver(llm, cfg.task_solver_config, device)
        self.device = device

    def __call__(self, request, subtasks, multi_processing=False):
        return self.got.solve(request, subtasks, multi_processing)

    def to(self, device):
        self.got.to(device)
        return self


@app.post("/tog")
@pool.register(lambda: TaskSolverWrapper(args.device))
async def tog(request: str = Form(...), subtasks: str = Form(...)):
    model = tog.__wrapped__.model
    output = model(request, subtasks)
    # return StreamingResponse(output)
    return JSONResponse(output)


@app.post("/task_decomposer")
@pool.register(lambda: TaskDecomposer(args.device, config.task_decomposer_cfg))
async def task_decomposer(request: str = Form(...)):
    model = task_decomposer.__wrapped__.model
    output = model(request)
    # return StreamingResponse(output)
    return JSONResponse(output)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
