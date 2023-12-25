from typing import Any, List

import copy
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from queue import PriorityQueue as PQ

import torch
import transformers
from langchain.schema import AIMessage, HumanMessage, SystemMessage


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def parse_response(response, pattern=None):
    if pattern is None:
        pattern = r"<Solution>((\S|\s)*)</Solution>"

    matched_group = re.search(pattern, response)
    parsed_response = None
    try:
        if matched_group:
            parsed_response = matched_group.group(1)

            parsed_response = json.loads(parsed_response)
        else:
            parsed_response = json.loads(response)
    except Exception as e:
        print(e)
        print(f"Error Response from LLM: {response}")
    return parsed_response


def send_message(llm, input, pattern=None, cnt=5):
    t = cnt
    while t > 0:
        response = llm(input)
        try:
            parsed_response = parse_response(response, pattern)
            break
        except Exception as e:
            print(e)
            print("Error: Can not parse the response from llm.")
            if response is not None:
                print(f"Response: {response}")
            t -= 1
    if t == 0:
        raise RuntimeError(
            f"Error: try {cnt} times but still can not parse the response from llm."
        )
    return response, parsed_response


def async_send_message(llm, inputs, pattern=None, cnt=5, max_workers=10):
    results = []
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for single_input in inputs:
            results.append(
                executor.submit(send_message, llm, single_input, pattern, cnt)
            )

    for i, r in enumerate(results):
        results[i] = r.result()
    return results


def args2str(arguments):
    arguments_str = ""
    for i, arg in enumerate(arguments):
        if "description" in arg:
            arguments_str += f'\t{arg["name"]} ({arg["type"]}): {arg["description"]}'
        else:
            arguments_str += f'\t{arg["name"]}: {arg["type"]}'
        if i != len(arguments) - 1:
            arguments_str += "\n"
    return arguments_str


def _exec(func, arguments_list):
    results = []
    for arguments in arguments_list:
        results.append(func(*arguments))

    return results


def _async_exec(func, arguments_list, max_workers=10, await_result=False):
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for arguments in arguments_list:
            results.append(executor.submit(func, *arguments))

    if not await_result:
        return results

    for i, r in enumerate(results):
        results[i] = r.result()

    return results


def batch_exec(func, arguments_list, max_workers=10, await_result=False):
    if max_workers == 0:
        return _exec(func, arguments_list)

    return _async_exec(func, arguments_list, max_workers, await_result)


class TaskDecomposer:
    def __init__(self, device, cfg):
        self.device = device
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            cfg.model,
            device_map=device,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.model,
            padding_side="left",
            use_fast=False,
        )
        self.smart_tokenizer_and_embedding_resize()

    def smart_tokenizer_and_embedding_resize(self):
        """Resize tokenizer and embedding.

        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        """
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"
        special_tokens_dict = dict()
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.model.get_input_embeddings().weight.data
            output_embeddings = self.model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    @torch.no_grad()
    def generate(self, user_input):
        prompt_no_input = PROMPT_DICT["prompt_no_input"]
        user_input = prompt_no_input.format_map({"instruction": user_input})
        input_ids = self.tokenizer(
            user_input,
            padding="longest",
            return_token_type_ids=False,
            return_tensors="pt",
        )
        input_ids = input_ids.to("cuda:0")
        generate_ids = self.model.generate(
            **input_ids,
            do_sample=True,
            max_length=self.tokenizer.model_max_length,
            top_k=30,
            top_p=0.85,
            temperature=0.1,
            repetition_penalty=1.0,
            eos_token_id=2,
            bos_token_id=1,
            pad_token_id=0,
        )
        result = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        result = result[len(user_input) :]
        return result

    def _convert_dict(self, result):
        if len(result) == 0:
            return result
        if "type" in result[0]["args"][0]:
            return result

        for subtask in result:
            new_args = []
            args = subtask["args"]
            for arg in args:
                new_args.append(copy.deepcopy(arg))
            subtask["args"] = new_args
            new_returns = []
            returns = subtask["returns"]
            for ret in returns:
                new_returns.append(copy.deepcopy(ret))
            subtask["returns"] = new_returns
        return result

    def solve(self, request, **kwargs) -> str:
        result = self.generate(request)
        result = parse_response(result)
        result = self._convert_dict(result)
        torch.cuda.empty_cache()
        return json.dumps(result, indent=2, ensure_ascii=False)

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return self.solve(*args, **kwargs)

    def to(self, device):
        self.model.to(device)
        return self.model


class SolutionExpert:
    def __init__(self, llm, cfg):
        self.llm = llm
        self.cfg = cfg
        self.prompts = cfg.prompts

    def solve(self, solutions, task_json, user_request):
        if len(solutions) == 0:
            return []
        if len(solutions) > 15:
            solutions = solutions[:15]

        arguments_list = []
        for i, s in enumerate(solutions):
            system_global_prompt = self.prompts.score_solution_system_prompt
            system_request_prompt = self.prompts.score_solution_request_prompt
            task_description = task_json["description"]
            solution_str = json.dumps(s)
            system_request_prompt = system_request_prompt.replace(
                "{{task}}", task_description
            )
            system_request_prompt = system_request_prompt.replace(
                "{{request}}", user_request
            )
            system_request_prompt = system_request_prompt.replace(
                "{{solution}}", solution_str
            )
            system_global_prompt = SystemMessage(content=system_global_prompt)
            system_request_prompt = SystemMessage(content=system_request_prompt)
            arguments_list.append(
                [self.llm, [system_global_prompt, system_request_prompt]]
            )
        results = batch_exec(
            send_message,
            arguments_list,
            min(15, len(arguments_list)),
            await_result=True,
        )
        pq = PQ()
        for i, (_, res) in enumerate(results):
            score = res.get("Score", 3)
            pq.put([-score, i])

        top = pq.get()
        optimal_solutions = [{"score": -top[0], "solution": solutions[top[1]]}]
        while not pq.empty():
            item = pq.get()
            score = -item[0]
            if score >= 3:
                optimal_solutions.append(
                    {"score": score, "solution": solutions[item[1]]}
                )
            if len(optimal_solutions) >= 5:
                break

        if len(optimal_solutions) == 0:
            return []
        return optimal_solutions


class ResourceExpert:
    def __init__(self, llm, cfg):
        self.llm = llm
        self.cfg = cfg
        self.prompts = cfg.prompts

    def solve(self, action_path, task, user_request):
        system_global_msg = SystemMessage(
            content=self.prompts.system_resource_global_prompt
        )
        task_args = copy.deepcopy(task["args"])
        for arg in task_args:
            arg["description"] = "It is provided by human."
        resources = json.dumps(task_args, indent=4)
        input_collection = []
        updated_action_path = []
        for action in action_path:
            black = "______"
            incompleted_arguments = []
            has_absent_arg = False
            for arg in action["args"]:
                arg_val = arg.get("value", None)
                arg_dict = {}
                if arg_val:
                    arg_dict[f"{arg['name']} ({arg['type']})"] = arg_val
                else:
                    arg_dict[f"{arg['name']} ({arg['type']})"] = black
                    has_absent_arg = True

                incompleted_arguments.append(arg_dict)

            if not has_absent_arg:
                resources = json.loads(resources)
                resources.extend(action["returns"])
                resources = json.dumps(resources, indent=4)
                updated_action_path.append(copy.deepcopy(action))
                continue

            input = json.dumps(incompleted_arguments)
            arguments_str = args2str(action["args"])
            tool_name = action["tool_name"]
            tool_description = action["description"]
            task_description = task["description"]
            returns_str = args2str(action["returns"])
            system_resource_prompt = copy.deepcopy(self.prompts.system_resource_prompt)
            system_resource_prompt = system_resource_prompt.replace(
                "{{request}}", user_request
            )
            system_resource_prompt = system_resource_prompt.replace(
                "{{task_description}}", task_description
            )
            system_resource_prompt = system_resource_prompt.replace(
                "{{resources}}", resources
            )
            system_resource_prompt = system_resource_prompt.replace(
                "{{tool_name}}", tool_name
            )
            system_resource_prompt = system_resource_prompt.replace(
                "{{tool_description}}", tool_description
            )
            system_resource_prompt = system_resource_prompt.replace(
                "{{arguments}}", arguments_str
            )
            system_resource_prompt = system_resource_prompt.replace(
                "{{returns}}", returns_str
            )
            system_resource_prompt = system_resource_prompt.replace("{{input}}", input)
            returns = action["returns"]
            for ret in returns:
                ret["description"] = f"It is generated by tool `{tool_name}`"

            resources = json.loads(resources)
            resources.extend(action["returns"])
            resources = json.dumps(resources, indent=4)
            system_request_msg = SystemMessage(content=system_resource_prompt)
            input_collection.append([system_global_msg, system_request_msg])
            updated_action_path.append(None)

        results = async_send_message(self.llm, input_collection)
        j = 0
        for i, action in enumerate(action_path):
            if updated_action_path[i] is not None:
                continue
            _, searched_args = results[j]
            j += 1
            updated_action = copy.deepcopy(action)
            for k, arg in enumerate(updated_action["args"]):
                arg_val = arg.get("value", None)
                if arg_val is None:
                    arg["value"] = list(searched_args[k].values())[0]

            updated_action_path[i] = updated_action
        return updated_action_path


class ThoughtsOnGraph:
    def __init__(self, llm, config):
        self.llm = llm
        self.config = config
        self.strategy = config.strategy
        self.prompts = config.prompts
        self.step = 0
        self.load_functions(config.tools)
        self._build_tool_conflict_dict()

    def load_functions(self, tools):
        self.tool_dict = {}
        self.resource_type = set()
        graph = {}
        if isinstance(tools, str) and os.path.exists(tools):
            tools = json.load(open(tools, "r"))

        self.tools = tools
        tools_wo_input = []
        for tool in self.tools:
            tool_name = tool["name"]
            if len(tool["args"]) == 0 or tool["args"][0]["type"] == "none":
                tools_wo_input.append(tool_name)

            for arg in tool["args"]:
                arg_type = arg["type"].split(".")[-1]
                arg["type"] = arg_type
                self.resource_type.add(arg_type)

            for ret in tool["returns"]:
                ret_type = ret["type"].split(".")[-1]
                ret["type"] = ret_type
                self.resource_type.add(ret_type)

            self.tool_dict[tool_name] = copy.deepcopy(tool)
            self._update_graph(graph, tool_name, tool)

        # Any resource nodes can be used as input of tools without input.
        for t in self.resource_type:
            graph[t]["children"].extend(tools_wo_input)

        self.graph = graph

    def _update_graph(self, graph, tool_name, info):
        """graph contains two types of nodes
        <resource node>:
            - identified by its type
            - its children are all <tool node> that contains the resource node as input arg
        <tool node>:
            - identified by its name
            - its chilren are all <resource node> it returns
        """
        if graph.get(tool_name, None) is None:
            graph[tool_name] = {
                "type": "tool",
                "children": [],
                # 'dependencies': [],
            }
        for arg in info["args"]:
            arg_type = arg["type"]
            if graph.get(arg_type, None) is None:
                graph[arg_type] = {"type": "resource", "children": []}

            graph[arg_type]["children"].append(tool_name)

            for ret in info["returns"]:
                ret_type = ret["type"]
                if graph.get(ret_type, None) is None:
                    graph[ret_type] = {"type": "resource", "children": []}

        graph[tool_name]["children"].extend(info["returns"])

    # @lru_cache(maxsize=32)
    def _assess_tool(self, user_request, task_description, tool):
        solution_assessment_msg = SystemMessage(
            content=self.prompts.tool_assessment_prompt
        )
        prompt = """User Request: \n{{user_request}}\n\nTask description: \n{{task}}\n\nHere is the description of the tool `{{tool_name}}`:\n {{tool_name}}: {{tool_description}}\nArgs: {{arguments}}\nReturns: {{returns}}\n\nThe above information may be useful for AI to make decision. Please refer to the scoring criteria and score the tool `{{tool_name}}` for this task. Notice that If the tool description contains keywords from the task description or it is hard to make decision, the score of this tool should be greater than or equal to 3."""
        arguments_str = args2str(tool["args"])
        returns_str = args2str(tool["returns"])
        prompt = prompt.replace("{{task}}", task_description)
        prompt = prompt.replace("{{tool_description}}", tool["description"])
        prompt = prompt.replace("{{tool_name}}", tool["name"])
        prompt = prompt.replace("{{arguments}}", arguments_str)
        prompt = prompt.replace("{{returns}}", returns_str)
        request = SystemMessage(content=prompt)
        cnt = 5
        while cnt > 0:
            raw_response, response = send_message(
                self.llm, [solution_assessment_msg, request]
            )
            try:
                if isinstance(response, list):
                    response = response[0]
                score = response["Score"]
                break
            except Exception:
                thought = response.get("Thought", None)
                if thought and thought[-2].isnumeric():
                    response["Score"] = int(thought[-2])
                else:
                    # if failed, just set a high score to avoid no solution.
                    response["Score"] = 4
                cnt -= 1

        if cnt == 0:
            response["Score"] = 4
        return response["Score"]

    def select_tools(self, tool_candidates, score_map):
        tool_score_list = []
        for i, tool in enumerate(tool_candidates):
            tool_name = tool["tool_name"]
            score = score_map.get(tool_name, None)
            if not isinstance(score, int):
                score = 3
                score_map[tool_name] = score
            tool_score_list.append((score, i, tool_name))
            tool_score_list = sorted(tool_score_list, reverse=True)
        if self.strategy == "greedy":
            return [
                tool_candidates[tool_score_list[0][1]],
            ]
        elif self.strategy == "beam":
            new_tool_candidates = [
                tool_candidates[item[1]]
                for item in tool_score_list[: min(5, len(tool_candidates))]
            ]
            # print(f'beam: {new_tool_candidates}')
            return new_tool_candidates
        elif self.strategy == "adaptive":
            new_tool_candidates = []
            for tool in tool_candidates:
                tool_name = tool["tool_name"]
                score = score_map[tool_name]
                if score >= 3:
                    new_tool_candidates.append(tool)
            if len(new_tool_candidates) == 0:
                new_tool_candidates = [
                    tool_candidates[tool_score_list[0][1]],
                ]
            # print(f'adaptive: {new_tool_candidates}')
            return new_tool_candidates
        else:
            raise NotImplementedError()

    def assess_tool(
        self,
        tool_candidates,
        user_request,
        task,
        tool_score_cache,
        multi_processing=True,
    ):
        if self.strategy == "exhaustive" or len(tool_candidates) == 0:
            return tool_candidates

        arguments_list = []
        tool_list = []
        for tool in tool_candidates:
            if not self.tool_dict[tool["tool_name"]]["domain"].endswith("general"):
                if tool["tool_name"] in tool_score_cache:
                    continue

                task_description = task["description"]
                arguments_list.append(
                    [user_request, task_description, self.tool_dict[tool["tool_name"]]]
                )
                tool_list.append(tool["tool_name"])
            else:
                tool_score_cache[tool["tool_name"]] = 4

        max_workers = 0
        if multi_processing:
            max_workers = min(20, len(arguments_list))
        results = batch_exec(self._assess_tool, arguments_list, max_workers)

        for r, t in zip(results, tool_list):
            if not isinstance(r, int):
                r = r.result()
            tool_score_cache[t] = r

        new_tool_candidates = self.select_tools(tool_candidates, tool_score_cache)
        return new_tool_candidates

    def _build_tool_conflict_dict(
        self,
    ):
        tool_conflict_list = [
            [
                "image_cropping",
                "highlight_object_on_image",
                "image_inpainting",
                "image_matting",
                "text_image_editing",
                "partial_image_editing",
            ],
            [
                "image_classification",
                "image_instance_segmentation",
                "image_segmentation_by_mask",
                "segment_anything",
                "object_detection",
                "visual_grounding",
                "text_image_editing",
            ],
            [
                "image_to_edge",
                "image_to_line",
                "image_to_hed",
                "image_to_scribble",
                "image_to_pose",
                "image_to_depth",
                "image_to_normal",
                "segment_anything",
                "image_instance_segmentation",
            ],
            [
                "question_answering",
                "image_question_answering",
                "image_captioning",
            ],
            [
                "select_category",
                "count_objects",
                "count_masks",
                "count_categories",
                "text_to_image",
                "optical_character_recognition",
                "edge_text_to_image",
                "pose_text_to_image",
                "normal_text_to_image",
                "scribble_text_to_image",
                "hed_text_to_image",
                "line_text_to_image",
                "segmentation_text_to_image",
                "depth_text_to_image",
            ],
            [
                "openai_chat_model",
                "text_to_text_generation",
            ],
        ]
        self.tool_conflict_dict = {}
        for tool_list in tool_conflict_list:
            for tool in tool_list:
                if tool not in self.tool_conflict_dict:
                    self.tool_conflict_dict[tool] = set(tool_list)
                else:
                    self.tool_conflict_dict[tool] |= set(tool_list)

    def _filter_tools(self, tools, action_path):
        tool_names = set([item["tool_name"] for item in action_path])
        new_tools = []
        for i, tool in enumerate(tools):
            if (
                tool["tool_name"] == "text_image_editing"
                and len(tool["args"][1].get("value", "").split()) < 3
            ):
                continue
            if (
                tool["tool_name"].startswith("select_")
                and len(tool["args"][1].get("value", "").split()) > 3
            ):
                continue
            if (
                len(action_path) > 0
                and action_path[-1]["tool_name"] == "image_instance_segmentation"
                and tool["tool_name"] == "partial_image_editing"
            ):
                continue
            if tool["tool_name"] not in self.tool_conflict_dict:
                new_tools.append(tool)
                continue

            conflict = self.tool_conflict_dict[tool["tool_name"]]
            if len(conflict - tool_names) < len(conflict):
                continue

            new_tools.append(tool)
        return new_tools

    def check_conflict(self, tool, input_arg):
        tool_name = tool["name"]
        input_arg_val = input_arg["value"]
        if tool_name.startswith("select_") and (
            "image_captioning" in input_arg_val
            or "image_question_answering" in input_arg_val
            or "count_" in input_arg_val
        ):
            return False
        if "question_answering" in tool_name and (
            "image_captioning" in input_arg_val
            or "image_question_answering" in input_arg_val
            or "count_" in input_arg_val
        ):
            return False

        return True

    def _find_args(self, input_args, tool):
        tool_args = copy.deepcopy(tool["args"])
        is_valid_solution = True

        for t_arg in tool_args:
            # if arg already have value, continue
            if t_arg.get("value", None) is not None:
                continue

            # check if this arg is possible to be filled by input args
            target_args = [arg for arg in input_args if t_arg["type"] == arg["type"]]

            # TODO: why only check for single candidates?
            # If there is only one candiate input arg for this tool arg.
            # check if confilct, if it is, this tool should be discard.
            if len(target_args) == 1:
                if self.check_conflict(tool, target_args[0]):
                    t_arg["value"] = target_args[0]["value"]
                else:
                    is_valid_solution = False
                    break

        return tool_args if is_valid_solution else None

    def _find_tool_candidates(self, input_args, used_tools, task_group):
        tool_candidates = []
        for input_arg in input_args:
            input_arg_type = input_arg["type"]
            input_arg_val = input_arg["value"]

            # traverse all tool that can accept the inputs of this subtask
            for child in self.graph[input_arg_type]["children"]:
                # no tools can be used twice
                if child in used_tools:
                    continue

                # some tools are impossible to be called with
                # the inputs generated by some other tools
                tool = copy.deepcopy(self.tool_dict[child])
                if not self.check_conflict(tool, input_arg):
                    continue

                if isinstance(task_group, str):
                    task_group = [
                        task_group,
                    ]

                skip_flag = True

                # if this tool is not in the subtask domain, skip it.
                for t in task_group:
                    # general tools can be used in any subtask
                    if t in tool["domain"] or tool["domain"].endswith("general"):
                        skip_flag = False
                        break

                # if we don't have enough resources for the inputs of the current tool.
                # skip this tool
                tool_args = tool["args"]
                current_resource_types = [res["type"] for res in input_args]
                for arg in tool_args:
                    if arg["type"] not in current_resource_types:
                        skip_flag = True
                        break

                if skip_flag:
                    continue

                # assign value to the input args of current tool by matching the type.
                for arg in tool_args:
                    if arg["type"] == input_arg_type:
                        arg["value"] = input_arg_val

                # check if it is possible to fill the remaining args
                # if not, skip to next tool
                tool_args = self._find_args(input_args, tool)
                if tool_args is None:
                    continue

                # add this tool as a candiate
                returns = tool["returns"]
                for i, ret in enumerate(returns):
                    ret["value"] = f'<TOOL-GENERATED>-{child}-{ret["type"]}-{i}'

                description = self.tool_dict[child]["description"]
                tool_candidates.append(
                    {
                        "tool_name": child,
                        "description": description,
                        "domain": self.tool_dict[child]["domain"],
                        "args": tool_args,
                        "returns": returns,
                    }
                )
        return tool_candidates

    def arg_dfs_search(
        self,
        idx,
        potential_target_resources,  # returns of current tool
        target_resources,  # the final outputs we need
        target_resources_candidates,
    ):
        if idx == len(potential_target_resources):
            target_resources_candidates.append(
                [potential_target_resources, target_resources]
            )
            return

        # TODO: hard to understand, better refactor to use backtrack style
        new_target_resources = copy.deepcopy(target_resources)
        new_potential_target_resources = copy.deepcopy(potential_target_resources)

        whether_is_target = False
        # traverse all final output and find if there is one could be filled by
        # current returns at idx position.
        for r in new_target_resources:
            if (
                not r.get("exists", False)
                and r["type"] == new_potential_target_resources[idx]["type"]
            ):
                new_potential_target_resources[idx]["value"] = r["value"]
                r["exists"] = True
                whether_is_target = True
                break

        if whether_is_target:
            self.arg_dfs_search(
                idx + 1,
                new_potential_target_resources,
                new_target_resources,
                target_resources_candidates,
            )

        self.arg_dfs_search(
            idx + 1,
            potential_target_resources,
            target_resources,
            target_resources_candidates,
        )

    def dfs_search(
        self,
        graph,
        user_request,
        input_args,
        target_resources,
        used_tools,
        action_path: List[dict],
        solutions,
        task,
        tool_score_cache,
        multi_processing=True,
    ):
        if len(action_path) > 5:
            return

        input_args = copy.deepcopy(input_args)
        task_group = task["task"]
        end_flag = all([res.get("exists", False) for res in target_resources])

        if end_flag:
            solutions.append(copy.deepcopy(action_path))
            return

        tool_candidates = self._find_tool_candidates(input_args, used_tools, task_group)
        tool_candidates = self._filter_tools(tool_candidates, action_path)

        # evaluate if tools are relavent to the given task
        tool_candidates = self.assess_tool(
            tool_candidates, user_request, task, tool_score_cache, multi_processing
        )

        for tool in tool_candidates:
            self.step += 1
            # figures out what resources remain to be generated.
            target_resources_candidates = []
            self.arg_dfs_search(
                0,
                copy.deepcopy(tool["returns"]),
                copy.deepcopy(target_resources),
                target_resources_candidates,
            )

            # with the new target resources, dfs to search other tools to fill out the resources.
            used_tools.append(tool["tool_name"])
            for (
                returns,
                new_target_resources,
            ) in target_resources_candidates:
                tool["returns"] = returns
                new_input_args = copy.deepcopy(input_args)
                new_input_args.extend(returns)
                action_path.append(tool)
                self.dfs_search(
                    graph,
                    user_request,
                    new_input_args,
                    new_target_resources,
                    used_tools,
                    action_path,
                    solutions,
                    task,
                    tool_score_cache,
                )
                action_path.pop(-1)
            used_tools.pop(-1)

        return

    def search(self, task, user_request, multi_processing=True):
        args = copy.deepcopy(task["args"])
        results = []
        action_path = []
        self.dfs_search(
            self.graph,
            user_request,
            args,
            copy.deepcopy(task["returns"]),
            [],
            action_path,
            results,
            task,
            {},
            multi_processing,
        )
        return results

    def _filter_solutions(self, solutions, task_json):
        """
        check:
            1)whether some generated arguments are used in subsequent tools;
            2) whether some input arguments are not used.
        """

        task_args = task_json["args"]
        new_solutions = []
        solution_set = set()
        for solution in solutions:
            solution_str = json.dumps(solution)
            if solution_str in solution_set:
                continue

            solution_set.add(solution_str)
            solution_args = []
            is_valid = True
            for tool in solution[::-1]:
                returns = tool["returns"]
                for ret in returns:
                    can_be_used = False
                    for s_arg in solution_args:
                        if s_arg["type"] != ret["type"]:
                            continue

                        s_val = s_arg.get("value", None)
                        ret_val = ret["value"]
                        if s_val is None or s_val == ret_val:
                            can_be_used = True
                            if s_val is None:
                                solution_args.remove(s_arg)
                            break
                    if len(solution_args) == 0:
                        can_be_used = True

                    if not can_be_used:
                        is_valid = False
                        break

                if not is_valid:
                    break
                solution_args.extend(tool["args"])

            if not is_valid:
                continue

            for t_arg in task_args:
                can_be_used = False
                for s_arg in solution_args:
                    if s_arg["type"] != t_arg["type"]:
                        continue

                    s_val = s_arg.get("value", None)
                    t_val = t_arg["value"]
                    if s_val is None or s_val == t_val:
                        can_be_used = True
                        if s_val is None:
                            solution_args.remove(s_arg)
                        break
                if not can_be_used:
                    is_valid = False
                    break

            if is_valid:
                new_solutions.append(solution)

        return new_solutions

    def infer_args(self, solutions):
        """
        For argument `arg` whose type is `bbox`, if it is not used by the subsequent tools
        and only one subsequent tool needs `arg` with the type of bbox,
        then we can directly set `arg` as the input argument for this tool.
        """
        solutions = copy.deepcopy(solutions)
        for solution in solutions:
            used_args = []
            blank_args = []
            for tool in solution[::-1]:
                tool_args = tool["args"]
                returns = tool["returns"]
                for ret in returns:
                    if ret["value"] in used_args:
                        continue

                    blank_args = [
                        arg for arg in blank_args if arg["type"] == ret["type"]
                    ]

                    if len(blank_args) != 1:
                        continue

                    blank_args[0]["value"] = ret["value"]
                    blank_args.remove(blank_args[0])

                for t_arg in tool_args:
                    if t_arg.get("value", None):
                        used_args.append(t_arg["value"])
                    else:
                        blank_args.append(t_arg)

        return solutions

    def solve(self, tasks, user_request, multi_processing=True):
        solutions_list = []
        for sub_task in tasks:
            solutions = self.search(sub_task, user_request, multi_processing)
            solutions = self._filter_solutions(solutions, sub_task)
            solutions = self.infer_args(solutions)
            solutions = self._filter_solutions(solutions, sub_task)
            solutions = sorted(solutions, key=lambda x: len(x))
            solutions_list.append(solutions)

        return solutions_list


class TaskSolver:
    def __init__(self, llm, config, device="cpu"):
        self.llm = llm
        self.config = config
        self.tog = ThoughtsOnGraph(llm, config.tog_cfg)
        self.solution_expert = SolutionExpert(llm, config.solution_expert_cfg)
        self.resource_expert = ResourceExpert(llm, config.resource_expert_cfg)
        self.device = device

    def fill_arguments_for_one_task(
        self, solutions, task_json, request, multi_processing=True
    ):
        max_workers = 0
        if multi_processing:
            max_workers = min(5, len(solutions))

        if len(solutions) == 0:
            return solutions

        arguments_list = []
        for i, item in enumerate(solutions):
            score = item["score"]
            solution = item["solution"]
            arguments_list.append([solution, task_json, request])

        exec_results = batch_exec(
            self.resource_expert.solve,
            arguments_list,
            max_workers,
            await_result=True,
        )
        results = []
        for i, item in enumerate(solutions):
            score = item["score"]
            solution = item["solution"]
            results.append({"score": score, "solution": exec_results[i]})

        return results

    def solve(
        self,
        request: str,
        task_json: str | dict,
        multi_processing: bool = False,
        **kwargs,
    ) -> str:
        if isinstance(task_json, str):
            task_json = json.loads(task_json)
        max_workers = 0
        if multi_processing:
            max_workers = 10
        solutions_list = self.tog.solve(task_json, request, multi_processing)

        arguments_list = []
        for i in range(len(task_json)):
            arguments_list.append(
                (
                    solutions_list[i],
                    task_json[i],
                    request,
                )
            )

        optimal_solutions_list = batch_exec(
            self.solution_expert.solve,
            arguments_list,
            min(max_workers, len(arguments_list)),
            await_result=True,
        )
        arguments_list = []
        for i in range(len(task_json)):
            arguments_list.append(
                [
                    optimal_solutions_list[i],
                    task_json[i],
                    request,
                    multi_processing,
                ]
            )

        final_solutions_list = batch_exec(
            self.fill_arguments_for_one_task,
            arguments_list,
            max_workers=min(max_workers, len(arguments_list)),
            await_result=True,
        )
        solutions_str = json.dumps(final_solutions_list, indent=2, ensure_ascii=False)
        return solutions_str

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return self.solve(*args, **kwargs)

    def to(self, device):
        pass
