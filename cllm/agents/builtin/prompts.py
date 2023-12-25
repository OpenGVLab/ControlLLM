RUN_PROMPT_TEMPLATE = """
I will ask you to perform a task, your job is to come up with a series of simple commands in Python that will perform the task.
To help you, I will give you access to a set of tools that you can use. Each tool is a Python function and has a docstring explaining the task it performs, the inputs it expects and the outputs it returns.
You should first explain which tool you will use to perform the task and for what reason, then write the code in Python.
Each instruction in Python should be a simple assignment. 

Tools:
<<tools>>

Task: "How are you."

I will use `question_answering` to answer the given question.

Answer:
```
output = question_answering(text=_task_)
```

Task: "describe `image_1`."

I will use `image_captioning` to answer the question on `image_1`.

Answer:
```
output = image_captioning(image=image_1)
```

Task: "generate an image with a dog."

I will use `text_to_image` to generate an image.

Answer:
```
output = text_to_image(text="an image with a dog")
```

Task: "<<prompt>>"

"""

CODE_PROMPT_TEMPLATE = '''
I will ask you to perform a task, your job is to come up with a series of simple commands in Python that will perform the task.
- To help you, I will give you access to a set of tools that you can use. 
- Each tool is a Python function and has a docstring explaining the task it performs, the inputs it expects and the outputs it returns.
- You should first explain which tool you will use to perform the task and for what reason, then write the code in Python.
- Each instruction in Python should be a simple assignment. 

Tools:
<<tools>>

=======

History:
```
# Task: "How are you."
```

I will use `question_answering` to answer the given question.

Answer:
```
# Task: "How are you."
output = question_answering(text=_task_)
```

=======

History:
```
# Task: "describe the given image."
```

I will use `image_question_answering` to answer the question on the input image.

Answer:
```
# Task: "describe the given image."
output = image_question_answering(text=_task_, image=image)
```

=======

History:
```
output = text_to_image(text="an image with a dog")
# Task: "describe the given image."
```

I will use `image_question_answering` to answer the question on the input image.

Answer:
```
# Task: "describe the given image."
output = image_question_answering(text=_task_, image=output)
```

=======

History:
```
# Task: "generate an image with a dog."
```

I will use `text_to_image` to generate an image.

Answer:
```
# Task: "generate an image with a dog."
output = text_to_image(text="an image with a dog")
```

=======

History:
```
<<history>>
# Task: "<<prompt>>"
```


'''
