import os
import time
import json
import uuid


class Logger:
    def __init__(self, device):
        self.device = device

    def __call__(
        self,
        history_msgs: list,
        task_decomposition: list,
        solution: list,
        record: str,
        like: bool,
    ):
        os.makedirs("logs", exist_ok=True)
        print(f"solution: {solution}")
        print(f"solution: {type(solution)}")
        state = {
            "history": history_msgs,
            "task_decomposition": task_decomposition,
            "solution": solution,
            "record": record,
            "like": like,
        }
        file_name = f'logs/{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}_{str(uuid.uuid4())[:6]}.json'
        json.dump(state, open(file_name, "w"), indent=4)

    def to(self, device):
        return self
