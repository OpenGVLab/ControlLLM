import traceback
import logging
from typing import Tuple, List
import copy
from pathlib import Path
from cllm.agents import container
import json
from collections import OrderedDict

from cllm.agents.builtin import BUILTIN_PLANS, load_builtin_plans
from cllm.agents.container import auto_type
from cllm.agents.base import DataType, NON_FILE_TYPES

from .interpretor import Interpretor
from .planner import Planner
from .responser import generate_response

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, stream=True, interpretor_kwargs={}):
        self.stream = stream
        self.planner = Planner(self.stream)
        self.interpretor = Interpretor(**interpretor_kwargs)
        self.SHORTCUT = "**Using builtin shortcut solution.**"
        BUILTIN_PLANS.update(load_builtin_plans("builtin_plan.json"))
        logger.info(BUILTIN_PLANS)

    def plan(self, request: str, state: dict):
        logger.info(request)

        resource_memory = state.get("resources", {})
        raw_solution = None
        # shortcut for builtin plan
        for trigger_prompt, _ in BUILTIN_PLANS.items():
            if request == trigger_prompt:
                return self.SHORTCUT

        # dynamic execution
        if raw_solution is None:
            raw_solution = self.planner.plan(request, resource_memory)
        return raw_solution

    def parse_solution_from_stream(self, raw_solution):
        return self.planner.parse(raw_solution)

    def execute(self, raw_solution: str, state: dict):
        resource_memory = state.get("resources")
        request = state["request"]
        solution = None
        if raw_solution == self.SHORTCUT:
            for trigger_prompt, builtin_plan in BUILTIN_PLANS.items():
                if request == trigger_prompt:
                    solution = builtin_plan
                    solution = self._fill_args(solution, resource_memory)
        else:
            solution = self.planner.parse(raw_solution)

        if not solution:
            return None
        try:
            history_msgs = state.get("history_msgs")
            return self.interpretor.interpret(solution, history_msgs)
        except Exception as e:
            traceback.print_exc()
        return None

    def reply(self, executed_plan: dict, outputs: list, state: dict):
        error_response = [
            auto_type(
                "response",
                DataType.TEXT,
                "Sorry, I cannot understand your request due to an internal error.",
            )
        ]
        state = copy.deepcopy(state)
        if (
            executed_plan is None
            or len(executed_plan) == 0
            or outputs is None
            or len(outputs) == 0
        ):
            return error_response, state
        resources = state.get("resources", OrderedDict())
        for o in outputs:
            if isinstance(o, container.File):
                resources[str(o.filename)] = str(o.rtype)
        state["resources"] = resources
        response = generate_response(state["request"], executed_plan, outputs)
        if len(response) == 0:
            return error_response, state
        logger.info(response)
        return response, state

    def run(self, task: str, state: dict) -> Tuple[List, str]:
        try:
            return self._run(task, state)
        except:
            traceback.print_exc()
            logger.info(traceback.format_exc())
            return [
                auto_type(
                    "response",
                    DataType.TEXT,
                    "Sorry, I cannot understand your request due to an internal error.",
                )
            ], "Internal Error"

    def _run(self, task: str, state: dict) -> Tuple[List, str]:
        logger.info(task)
        BUILTIN_PLANS.update(load_builtin_plans("builtin_plan.json"))
        logger.info(BUILTIN_PLANS)
        resource_memory = state.get("resources", OrderedDict())
        history_msgs = state.get("history_msgs", [])
        plan = None

        # shortcut for builtin plan
        for trigger_prompt, builtin_plan in BUILTIN_PLANS.items():
            if task == trigger_prompt:
                plan = builtin_plan
                plan = self._fill_args(plan, resource_memory)

        # dynamic executation
        if plan is None:
            plan = self.planner.planning(task, resource_memory)
        logger.info(plan)

        executed_plan, output_files = self.interpretor.interpret(
            plan, resource_memory, history_msgs
        )
        logger.info(output_files)
        for o in output_files:
            if isinstance(o, container.File):
                resource_memory[o.filename] = str(o.rtype)

        outputs = generate_response(task, executed_plan, output_files)

        logger.info(outputs)
        return outputs, executed_plan

    def _fill_args(self, plan, memory):
        plan = copy.deepcopy(plan)
        latest_resource = OrderedDict()
        for key, val in memory.items():
            latest_resource[val] = key

        for actions in plan:
            for action in actions:
                for key, val in action.inputs.items():
                    if "<TOOL-GENERATED>" not in val:
                        action.inputs[key] = latest_resource.get(val, val)
        return plan
