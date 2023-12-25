from transformers import pipeline
from PIL import Image
import torch


class HuggingfacePipeline:
    def __init__(self, task, device="cpu", **kwargs):
        # dtype=None
        self.device = device
        self.task = task
        self.pipeline = pipeline(task, device=device, **kwargs)

    def __call__(self, *args, **kwargs):
        # print(f'HuggingfacePipeline. type(image): {type(image)}')
        output = self.pipeline(*args, **kwargs)
        # print(f'end HuggingfacePipeline. output: {output}')
        return output

    def to(self, device):
        self.pipeline.model.to(device=device)


class HuggingfacePipelineNLP:
    def __init__(self, task=None, device="cpu", **kwargs):
        # dtype=None
        self.device = device
        self.task = task
        self.model = pipeline(task, device=device, **kwargs)

    def __call__(self, text: str, *args, **kwargs):
        if self.task == "summarization":
            output = self.model(text, *args, **kwargs)
        elif self.task == "text2text-generation":
            output = self.model(text, *args, **kwargs)
        else:
            output = self.model(text, *args, **kwargs)
        if self.task in ["summarization", "text2text-generation"]:
            return list(output[0].values())[0]
        if self.task == "question-answering":
            return output["answer"]
        return output

    def to(self, device):
        self.model.model.to(device)
        return self
