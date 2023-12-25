
score_solution_system_prompt = """Given a task and a solution, The AI assistant needs to score the solution and respond in json format. Please notice that AI assistant should think. The AI assistant should pay more attention to relevance between the description of each tool in the solution and task.

The AI assistant respond with JSON format as follows: <Solution>{"Thought": "thought", "Score": score}</Solution>.

"Thought" filed records the model’s thinking process step by step within 80 words, which give the reasons why assistant gives this score.

"Score" filed denotes score that uses to assess whether this tool is useful for this task. "Score" is in [1, 2, 3, 4, 5]. Here is the scoring criteria: "Score"=1: The solution is totally not related to user's request and can not solve the task. "Score"=2: The solution is somewhat not related to user's request and may not solve the task. "Score"=3: The solution is probably related to the user's intention and may solve the task but it may not be the optimal one. "Score">3: The solution is closely or directly related to what the user wants and could satisfactorily solve the task. In a nut shell, the higher the score, the greater the likelihood of the solution solving the given task.

You should always respond in the following format:

<Solution> `SOLUTION` </Solution>

`SOLUTION` should strictly comply with JSON format described above."""

"""Given a task and a solution, The AI assistant needs to score the solution and respond in json format. Please notice that AI assistant should think. The AI assistant should pay more attention to relevance between the description of each tool in the solution and task.

The AI assistant respond with JSON format as follows: <Solution>{"Thought": "thought", "Score": score}</Solution>.

"Thought" filed records the model’s thinking process step by step within 80 words, which give the reasons why assistant gives this score.

"Score" filed denotes score that uses to assess whether this tool is useful for this task. "Score" is in [1, 2, 3, 4, 5]. Here is the scoring criteria: "Score"<3: The solution is basically not related to user's request and can not solve the task. "Score"=3: The solution is somewhat related to the user's intention and may solve the task but it may not be the optimal one. "Score">3: The solution is closely or directly related to what the user wants and could satisfactorily solve the task.

You should always respond in the following format:

<Solution> `SOLUTION` </Solution>

`SOLUTION` should strictly comply with JSON format described above."""

prompts = dict(
    score_solution_system_prompt=score_solution_system_prompt,
    score_solution_request_prompt='''User's request: "{{request}}"
Task description: "{{task}}".

Here is the description of the solution:
{{solution}}

Please refer to the scoring criteria and score this solution based on the task description. You should think carefully before scoring the solution. Notice that If the keywords in the solution are close in meaning to the keywords in the task description, then the score of this solution is at least 3.''',

    solution_selection_examples=[
        {
            "role": "user",
            "content": """User's request: [ what is it in sdf.png ].
        Here is the Task: [{"task_description": "detect the object in sdf.png", "task": "image-perception", "id": 0, "dep": [-1], "args": {"sdf.png": "image", "what is it in sdf.png": "text"}, "returns": {"<GENERATED>-0": "text"}}].
        User's request and task description may contain the information that is useful for AI to make decision.
        Here is the solution proposals to solve the task: [ {{solutions}} ]""",
        },
        {
            "role": "assistant",
            "content": "<Solution>[{\"task_description\": \"Generate an image of a mountain and animals.\", \"task\": [\"image-generation\"], \"id\": 0, \"dep\": [-1], \"args\": {\"Generate an image of a mountain and animals\": \"text\"}, \"returns\": {\"<GENERATED>-0\": \"image\"}}, {\"task_description\": \"Perform visual question-answering on the generated image to count the number of animals.\", \"task\": \"image-perception\", \"id\": 1, \"dep\": [0], \"args\": {\"<GENERATED>-0\": \"image\"}, \"returns\": {\"<GENERATED>-1\": \"text\"}}]</Solution>",
        },
    ],
)
