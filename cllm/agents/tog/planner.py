import json
from typing import List
import logging

from ..base import Action, NON_FILE_TYPES
from cllm.services.tog import TaskSolver, TaskDecomposer, config
from cllm.services.nlp.llms import ChatOpenAI, MessageMemory
from cllm.services.tog.api import tog, task_decomposer
from collections import OrderedDict
import copy


logger = logging.getLogger(__name__)


class Planner:
    def __init__(
        self, streaming=False, backend="remote", device="cuda:0", **llm_kwargs
    ):
        self.streaming = streaming
        if backend == "local":
            self.cfg = config
            self.device = device
            self.mem = MessageMemory(**self.cfg.memory)
            self.llm = ChatOpenAI(temperature=0.2, **llm_kwargs)
            self.tog = TaskSolver(self.llm, self.cfg.task_solver_config, device).solve
            self.decomposer = TaskDecomposer(device, self.cfg.task_decomposer_cfg).solve
        elif backend == "remote":
            self.decomposer = task_decomposer
            self.tog = tog
        else:
            raise ValueError("Backend should be chosen from [remote, local]")

    def _find_latest_resource(self, resources, type):
        for key, val in list(resources.items())[::-1]:
            if val == type:
                return key
        return None

    def _check_task_decomposition(
        self, task_decomposition: str | list, available_resources: dict
    ):
        copy_task_decomposition = copy.deepcopy(task_decomposition)
        available_resources = copy.deepcopy(available_resources)
        if isinstance(copy_task_decomposition, str):
            copy_task_decomposition = json.loads(copy_task_decomposition)

        for subtask in copy_task_decomposition:
            for arg in subtask["args"]:
                if arg["type"] in NON_FILE_TYPES:
                    continue

                r_type = available_resources.get(arg["value"], "None").split(".")[-1]
                if arg["value"] not in available_resources or arg["type"] != r_type:
                    new_value = self._find_latest_resource(
                        available_resources, arg["type"]
                    )
                    if new_value is None:
                        logger.error(
                            f"No available resource for {arg['value']} with type {arg['type']}"
                        )
                        return None

                    arg["value"] = new_value

            available_resources[subtask["returns"][0]["value"]] = subtask["returns"][0][
                "type"
            ]
        return json.dumps(copy_task_decomposition, indent=2, ensure_ascii=False)

    def wrap_request(self, request, memory):
        logger.info(memory)
        resource_list = {k: v.split(".")[-1] for k, v in memory.items()}
        request = f"Resource list: {resource_list}\n{request}"
        logger.info(f"Input: {request}")
        return request

    def solve_streaming(self, request: str, memory: dict = OrderedDict()):
        request = self.wrap_request(request, memory)
        sub_tasks = self.decomposer(request, streaming=self.streaming)
        logger.info(f"Task decomposition: \n{sub_tasks}")
        sub_tasks = self._check_task_decomposition(sub_tasks, memory)
        yield sub_tasks
        if sub_tasks in [None, "", []]:
            yield None
        else:
            solutions = self.tog(request, sub_tasks, streaming=self.streaming)
            yield solutions

    def solve(self, request: str, memory: dict = OrderedDict()) -> List:
        self.wrap_request(request, memory)
        sub_tasks = self.decomposer(request)
        solutions = self.tog(request, sub_tasks)
        print(f"solutions: {solutions}")
        return sub_tasks, solutions

    def plan(self, task, memory: dict = OrderedDict()) -> List:
        if self.streaming:
            return self.solve_streaming(task, memory)
        else:
            return self.solve(task, memory)

    def _check_solutions(self, solution: List | str) -> bool:
        if isinstance(solution, str):
            solution = json.loads(solution)
        if len(solution) == 0:
            return False

        valid = True
        for i, stage_candiate in enumerate(solution):
            if len(stage_candiate) == 0:
                logger.error(f"No solution is found in {i}-th subtask.")
                valid = False
            elif (
                "solution" in stage_candiate[0]
                and len(stage_candiate[0]["solution"]) == 0
            ):
                logger.error(f"No solution is found in {i+1}-th subtask.")
                valid = False
            else:
                logger.info(f"Solutions for {i+1}-th subtask:\n{stage_candiate}")
        return valid

    def parse(self, solution: List | str) -> List[List[Action]]:
        if isinstance(solution, str):
            solution = json.loads(solution)

        if not self._check_solutions(solution):
            return None

        if isinstance(solution[0], Action):
            return solution

        stages = []
        for i, stage_candiate in enumerate(solution):
            stage = stage_candiate[0]["solution"]
            actions = []
            for action in stage:
                inputs = {arg["name"]: arg["value"] for arg in action["args"]}
                outputs = [r["value"] for r in action["returns"]]
                actions.append(
                    Action(action["tool_name"], inputs=inputs, outputs=outputs)
                )
            stages.append(actions)
        return stages

    def __call__(
        self, request: str, memory: dict = OrderedDict()
    ) -> List[List[Action]]:
        solution = self.solve(request, memory)
        return self.parse(solution)
