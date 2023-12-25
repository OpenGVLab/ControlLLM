from cllm.services.tog.utils import build_tool_prompt
from cllm.agents.builtin.tools import GENERAL_TOOLS


def test():
    print(build_tool_prompt(GENERAL_TOOLS[0]))

    # This is a tool that select the target classes in category list with the given condition. It is commonly used to filter out the objects with the same type.
    # Args:
    #   category_list (category): the list to be processed
    #   condition (text): the condition to select objects
    # Returns
    #   list (list): the selected list


def generate_json():
    from cllm.agents.builtin.tools import TOOLS

    tools = []
    for tool in TOOLS.values():
        tool.description = build_tool_prompt(tool)
        tools.append(tool.dict())
    import json

    with open("tools.json", "w") as f:
        json.dump(tools, f, indent=4)


generate_json()
