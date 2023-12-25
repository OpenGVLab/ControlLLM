prompts = dict(
    task_decomposition_prompt="""The AI assistant is professional and parse user input to several tasks with lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know and ask user for help.

The AI assistant can parse user input to several tasks with JSON format as follows: <Solution>[{"description": task_description, "task": task_domain, "id": task_id, "dep": dependency_task_id, "args": {text or <GENERATED>-dep_id: "text", image_url or <GENERATED>-dep_id: "image", audio_url or <GENERATED>-dep_id: "audio"}, "returns":{<GENERATED>-task_id: "segmentation"}}]</Solution>.

The "description" should describe the task in details and AI assistant can add some details to improve the user’s request without changing the user’s original intention. The special tag "<GENERATED>-dep_id" refers to the one generated text/image/audio/video/segmentation/mask in the dependency task (Please consider whether the dependency task generates resources of this type.) and "dep_id" must be in "dep" list. The special tag "<GENERATED>-task_id" refers to the one generated text/image/audio/video/segmentation/mask in this task and "task_id" should be in line with field "id" of this task. The "dep" field denotes the ids of the previous prerequisite tasks which generate a new resource that the current task relies on. The "args" field and the "returns" field denote the input resources and output resources of this task, respectively. The type of resource must be in ["text", "image", "audio", "video", "segmentation", "mask"], nothing else. The "task" MUST be selected from the following options: "chat", "question-answering", "image-generation", "image-editing", "image-perception", "audio-perception", "audio-generation", "audio-editing", "video-perception", "video-generation", "video-editing", nothing else. Think step by step about all the tasks that can resolve the user's request.

Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. If some args is not found, you cannot assume that they already exist. You can think a new task to generate those args that do not exist or ask for user's help. If the user input can't be parsed, you need to reply empty JSON [].

You should always respond in the following format:

<Solution> `SOLUTION` </Solution>

`SOLUTION` should be strictly with JSON format described above.
""",
    user_request_prompt1='Now I input { {{input}} }. Pay attention to the "args" and "returns" types of tasks.',
    user_request_prompt2='The chat log [ {{context}} ] may contain the resources I mentioned. Now I input { {{input}} }. Pay attention to the "args" and "returns" types of tasks.',
    task_decomposition_examples=[
        {
            "role": "user",
            "content": "Generate a image describing a mountain and animals and tell me how many animals in this image.",
        },
        {
            "role": "assistant",
            "content": '<Solution>[{"description": "Generate an image of a mountain and animals.", "task": ["image-generation"], "id": 0, "dep": [-1], "args": {"Generate an image of a mountain and animals": "text"}, "returns": {"<GENERATED>-0": "image"}}, {"description": "Perform visual question-answering on the generated image to count the number of animals.", "task": "image-perception", "id": 1, "dep": [0], "args": {"<GENERATED>-0": "image"}, "returns": {"<GENERATED>-1": "text"}}]</Solution>',
        },
        {
            "role": "user",
            "content": "Give you some pictures e1.jpg, e2.png, e3.jpg, help me count the number of sheep?",
        },
        {
            "role": "assistant",
            "content": '<Solution>[{"description": "count the number of sheep for image e1.jpg", "task": "image-perception", "id": 0, "dep": [-1], "args": {"e1.jpg": "image"}, "returns": {"<GENERATED>-0": "text"}}, {"description": "count the number of sheep for image e2.png", "task": "image-perception", "id": 1, "dep": [-1], "args": {"e2.png": "image"}, "returns": {"<GENERATED>-1": "text"}},  {"description": "count the number of sheep for image e3.jpg", "task": "image-perception", "id": 2, "dep": [-1], "args": {"e3.jpg": "image"}, "returns": {"<GENERATED>-2": "text"}}]</Solution>',
        },
        {
            "role": "user",
            "content": "please show me a video and an image of (based on the text) 'a boy is running' and dub it",
        },
        {
            "role": "assistant",
            "content": '<Solution>[{"description": "generate a new video showing \'a boy is running\'", "task": "video-generation", "id": 0, "dep": [-1], "args": {"a boy is running": "text"}, "returns": {"<GENERATED>-0": "video"}}, {"description": "generate a new audio containing \'a boy is running\'", "task": "text-to-speech", "id": 1, "dep": [-1], "args": {"a boy is running": "text"}, "returns": {"<GENERATED>-1": "audio"}}, {"description": "generate a new image showing \'a boy is running\'", "task": "image-generation", "id": 2, "dep": [-1], "args": {"a boy is running": "text"}, "returns": {"<GENERATED>-2": "image"}}]</Solution>',
        },
        {"role": "user", "content": "detect the dog in eqads.png"},
        {
            "role": "assistant",
            "content": '<Solution>[{"description": "Detect the dog in eqads.png", "task": "image-perception", "id": 0, "dep": [-1], "args": {"eqads.png": "image"}, "returns": {"<GENERATED>-0": "image"}}]</Solution>',
        },
        {"role": "user", "content": "please show me a joke and an image of cat"},
        {
            "role": "assistant",
            "content": '<Solution>[{"description": "talk with user and tell user a joke", "task": "conversational", "id": 0, "dep": [-1], "args": {"please show me a joke": "text"}, "returns": {"<GENERATED>-0": "text"}}, {"description": "generate a new image of cat", "task": "image-generation", "id": 1, "dep": [-1], "args": {"a photo of cat": "text"}, "returns": {"<GENERATED>-1": "image"}}]</Solution>',
        },
        {"role": "user", "content": "generate a new image containing dog and cat"},
        {
            "role": "assistant",
            "content": '<Solution>[{"description": "generate a new image containing dog and cat", "task": "image-generation", "id": 0, "dep": [-1], "args": {"a photo of dog and cat": "text"}, "returns": {"<GENERATED>-0": "image"}}]</Solution>',
        },
        {"role": "user", "content": "generate a new image from segmentation a.png"},
        {
            "role": "assistant",
            "content": '<Solution>[{"description": "generate a new image containing dog and cat", "task": "image-generation", "id": 0, "dep": [-1], "args": {"a.png": "segmentation"}, "returns": {"<GENERATED>-0": "image"}}]</Solution>',
        },
        {"role": "user", "content": "generate the segmentation map of a.png"},
        {
            "role": "assistant",
            "content": '<Solution>[{"description": "segment a.png to obtain segmentation map of a.png", "task": "image-perception", "id": 0, "dep": [-1], "args": {"a.png": "image"}, "returns": {"<GENERATED>-0": "segmentation"}}]</Solution>',
        },
        {
            "role": "user",
            "content": "generate a new image from the canny detection result of aaa.png and it should contain some objects in bbb.jpg",
        },
        {
            "role": "assistant",
            "content": '<Solution>[{"task_description": "using canny operator to detect the edge of a.png", "task": "image-to-canny", "id": 0, "dep": [-1], "args": {"image": "aaa.png" }, "returns": {"mask": "<GENERATED>-0"}, {"task_description": "detect the objects in image bbb.jpg.", "task": "image-perception", "id": 1, "dep": [-1], "args": {"image": "bbb.jpg", "text": "What objects are in the picture?"}, "returns": {"text": "<GENERATED>-1"}, {"task_description": "generate a new image from canny detection <GENERATED>-0 and the new image contains <GENERATED>-1 in the bbb.jpg", "task": "image-generation", "id": 2, "dep": [0,1], "args": {"mask": "<GENERATED>-0", "text": "<GENERATED>-1" }, "returns": {"image": "<GENERATED>-2"}}]</Solution>',
        },
        {
            "role": "user",
            "content": "Remove all objects that appear in image a.png from image b.jpeg",
        },
        {
            "role": "assistant",
            "content": '<Solution>[{"description": "recognize the objects in a.png", "task": "image-perception", "id": 0, "dep": [-1], "args": {"a.png": "image"}, "returns": {"<GENERATED>-0": "text"}}, {"description": "segment the image b.jpeg by natural language", "task": "image-perception", "id": 1, "dep": [0], "args": {"b.jpeg": "image", "<GENERATED>-0": "text"}, "returns": {"<GENERATED>-1": "image"}}, {"description": "remove the objects by mask", "task": "image-editing", "id": 2, "dep": [1], "args": {"b.jpeg": "image", "<GENERATED>-1": "mask"}, "returns": {"<GENERATED>-2": "image"}}]</Solution>',
        },
        {
            "role": "user",
            "content": "I want to make dumpling, but I do not know how to make it. Can you help me?",
        },
        {
            "role": "assistant",
            "content": '<Solution>[{"description": "describe the recipe of dumpling", "task": "question-answering", "id": 0, "dep": [-1], "args": {"how to cook dumpling": "text"}, "returns": {"<GENERATED>-0": "text"}}]</Solution>',
        },
    ],
)

if __name__ == "__main__":
    import json
    import re

    new_examples = []
    for i, example in enumerate(prompts["task_decomposition_examples"]):
        if i % 2 == 0:
            new_example = {}
            new_example["role"] = example["role"]
            new_example["content"] = example["content"]
            new_examples.append(new_example)
            continue
        print(example["content"])
        content = example["content"]
        pattern = r"<Solution>(.*?)</Solution>"
        content = content.replace("\n", "")
        result = re.search(pattern, content)
        if result:
            result = result.group(1)
            print(f"result = {result}")
            task_json = json.loads(result)
        else:
            task_json = {}
        # print(f'result = {result}')
        # args = task_json['args']
        for task in task_json:
            args = task["args"]
            # args = {v: k for k, v in args.items()}
            task["args"] = args
            returns = task["returns"]
            # returns = {v: k for k, v in returns.items()}
            task["returns"] = returns

        args = json.dumps(args, sort_keys=False)
        # task_json['args'] = args
        task_json = json.dumps(task_json, sort_keys=False)
        new_example = {}
        new_example["role"] = "assistant"
        new_example["content"] = f"<Solution>{task_json}</Solution>"
        # example = '<Solution>(.*?)</Solution>'
        new_examples.append(new_example)

    new_examples = json.dump(
        new_examples,
        open(
            "examples.json",
            "w",
        ),
        indent=4,
    )
