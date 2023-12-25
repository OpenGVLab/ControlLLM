from cllm.agents import Tool


def build_tool_description(tool: Tool):
    description = tool.description
    if description.endswith('.'):
        description = description[:-1]
    args = [
        f'a `{arg.name}` in the type of {arg.type} represents the {arg.description}'
        for arg in tool.args
    ]
    args = ', and '.join(args)
    usage = tool.domain
    desc = f'This is a tool that {description}. It takes {args}. This tool is commonly used to {usage}.'
    return desc


def build_tool_prompt(tool: Tool):
    description = tool.description
    if description.endswith('.'):
        description = description[:-1]
    if len(tool.usages) == 0:
        usage = tool.domain
    else:
        usage = '\n'.join(['  ' + u for u in tool.usages])
    doc_string = 'Args:\n'
    for p in tool.args:
        doc_string += '  {} ({}): {}\n'.format(p.name, p.type, p.description)
    doc_string += 'Returns\n'

    for output in tool.returns:
        doc_string += '  {} ({}): {}\n'.format(
            output.name, output.type, output.description
        )

    desc = f'This is a tool that {description}. \nIt is commonly used as follows: \n{usage} \n{doc_string}'
    return desc
