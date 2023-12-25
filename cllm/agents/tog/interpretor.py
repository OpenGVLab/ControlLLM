import logging
from traceback import print_exc
from typing import List, Dict
import os.path as osp
import io
import copy
import re
import uuid
from matplotlib.pyplot import isinteractive

from numpy import isin
from cllm.agents.base import Action, DataType, Tool, NON_FILE_TYPES
from cllm.agents.builtin import TOOLS
from cllm.agents.container import auto_type
from cllm.utils import get_real_path, get_root_dir, transform_msgs

logger = logging.getLogger(__name__)


def code(source, type="py"):
    return f"```{type}\n{source}\n```"


class Interpretor:
    def __init__(self):
        self.tools = TOOLS
        self.non_file_types = NON_FILE_TYPES

    def interpret(self, stages: List[List[Action]], history_msgs: List = []):
        memory = {}
        solution = copy.deepcopy(stages)
        history_msgs = copy.deepcopy(history_msgs)
        history_msgs = transform_msgs(history_msgs)
        has_error = False
        for actions in solution:
            for action in actions:
                tool = self.load_tool(name=action.tool_name)
                tool_inputs = self.load_args(tool, action.inputs, memory)
                tool_inputs["history_msgs"] = history_msgs
                tool_inputs["root_dir"] = get_root_dir()
                try:
                    tool_outputs = tool.model(**tool_inputs)
                    action.inputs = self._update_inputs(memory, action.inputs)
                    action.outputs, wrapped_outputs = self._update_output(
                        memory, action, tool_outputs, tool
                    )
                    logger.info(
                        "Call {}, args {}, return {}".format(
                            action.tool_name, action.inputs, action.outputs
                        )
                    )
                    executed_action = (
                        action.tool_name,
                        action.inputs,
                        action.outputs,
                    )
                except FileNotFoundError as e:
                    print_exc()
                    tool_outputs = None
                    logger.error(f"Error when executing {action.tool_name}: {e}")
                    has_error = True
                    wrapped_outputs = []
                    executed_action = (
                        action.tool_name,
                        action.inputs,
                        f"FileNotFoundError: No such file or directory: {osp.basename(e.filename)}",
                    )
                except Exception as e:
                    print_exc()
                    tool_outputs = None
                    has_error = True
                    logger.error(f"Error when executing {action.tool_name}: {e}")
                    wrapped_outputs = []
                    executed_action = (
                        action.tool_name,
                        action.inputs,
                        f"Internal error: {e}",
                    )
                yield executed_action, solution, wrapped_outputs
                if has_error:
                    return

    def _update_output(self, memory, action, tool_outputs, tool):
        outputs = []
        wrapped_outputs = []
        if action.outputs is not None:
            if len(action.outputs) == 1:
                tool_outputs = [tool_outputs]
            for i, (arg_name, arg_value) in enumerate(
                zip(action.outputs, tool_outputs)
            ):
                memory[arg_name] = arg_value
                if arg_value is None or isinstance(arg_value, Exception):
                    outputs.append(arg_value)
                    arg_value = f"{arg_value}"
                    wrapped_outputs.append(
                        auto_type(
                            arg_name,
                            DataType.TEXT,
                            arg_value,
                        )
                    )
                    continue

                if isinstance(arg_value, (dict, list)):
                    arg_value = self.pretty_floats(arg_value)

                if tool.returns[i].type in self.non_file_types:
                    outputs.append(arg_value)
                    wrapped_outputs.append(
                        auto_type(
                            arg_name,
                            tool.returns[i].type,
                            arg_value,
                        )
                    )

                    continue

                transformed_output = self.transform_output(
                    action.inputs,
                    tool.name,
                    tool.args,
                    arg_value,
                    tool.returns[i].type,
                )

                outputs.append(transformed_output)
                memory[arg_name] = transformed_output
                if not isinstance(transformed_output, list):
                    wrapped_outputs.append(
                        auto_type(
                            arg_name,
                            tool.returns[i].type,
                            transformed_output,
                        )
                    )
                    continue

                for output in transformed_output:
                    if DataType.MASK == tool.returns[i].type:
                        output = output if isinstance(output, str) else output["mask"]
                    wrapped_outputs.append(
                        auto_type(
                            arg_name,
                            tool.returns[i].type,
                            output if isinstance(output, str) else output["mask"],
                        )
                    )
        return outputs, wrapped_outputs

    def pretty_floats(self, obj):
        if isinstance(obj, float):
            return round(obj, 4)
        elif isinstance(obj, dict):
            return dict((k, self.pretty_floats(v)) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            return list(map(self.pretty_floats, obj))
        return obj

    def _update_inputs(self, memory, action_inputs):
        action_inputs = copy.deepcopy(action_inputs)
        for key, value in action_inputs.items():
            if "<TOOL-GENERATED>" in value:
                action_inputs[key] = memory.get(value, value)
            elif "<GENERATED>" in value:
                action_inputs[key] = memory.get(value, value)

        return action_inputs

    def gen_filename(self, too_name, resource_type):
        def to_camelcase(s):
            res = re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), s)
            res = res[0].upper() + res[1:]
            return res

        if resource_type == DataType.VIDEO:
            ext = "mp4"
        elif resource_type == DataType.AUDIO:
            ext = "wav"
        elif resource_type == DataType.HTML:
            ext = "html"
        else:
            ext = "png"
        too_name = too_name.replace("_to_", "2_")
        too_name = to_camelcase(too_name)
        this_file_id = str(uuid.uuid4())[:6]
        type_str = str(resource_type).split(".")[-1]
        return f"{this_file_id}_{type_str}.{ext}"

    def _save_resource(self, file_name, resource, resource_type):
        if isinstance(resource, dict):
            if "mask" in resource:
                resource = resource["mask"]
        if resource_type == DataType.HTML:
            with open(get_real_path(file_name), "w") as fout:
                fout.write(resource)
        elif resource is not None:
            if isinstance(resource, io.BufferedReader):
                resource = resource.read()
            with open(get_real_path(file_name), "wb") as fout:
                fout.write(resource)
        else:
            return None

    def transform_output(
        self, action_inputs, tool_name, tool_args, tool_output, output_type
    ):
        if output_type != DataType.MASK:
            if isinstance(tool_output, list):
                results = []
                for output in tool_output:
                    file_name = self.gen_filename(tool_name, output_type)
                    self._save_resource(file_name, output, output_type)
                    results.append(file_name)
                return results
            else:
                file_name = self.gen_filename(tool_name, output_type)
                self._save_resource(file_name, tool_output, output_type)
                return file_name

        tool_output = copy.deepcopy(tool_output)
        if isinstance(tool_output, list):
            for output in tool_output:
                if isinstance(output["mask"], str):
                    continue

                file_name = self.gen_filename(tool_name, output_type)
                self._save_resource(file_name, output, output_type)
                output["mask"] = file_name
        elif isinstance(tool_output, bytes):
            file_name = self.gen_filename(tool_name, output_type)
            self._save_resource(file_name, tool_output, output_type)
            tool_output = file_name
        elif tool_output is None:
            pass
        else:
            raise RuntimeError("Wrong type.")

        return tool_output

    def load_tool(self, name):
        return self.tools[name]

    def load_args(self, tool: Tool, action_inputs, memory):
        real_args = {}
        for item in tool.args:
            arg_name = item.name
            arg_value = action_inputs[arg_name]
            if "<GENERATED>" in arg_value or "<TOOL-GENERATED>" in arg_value:
                assert arg_value in memory, print(f"Unknown {arg_name}: {arg_value}")
                real_args[arg_name] = memory[arg_value]
            else:
                real_args[arg_name] = arg_value
        return real_args

    @property
    def variables(self):
        return {k: v for k, v in self.memory.items() if k not in TOOLS and k != "print"}
