import torch
import requests
import json
from PIL import Image
from io import BytesIO
from pathlib import Path
from transformers import ViltProcessor, ViltForQuestionAnswering
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.conversation import conv_templates, SeparatorStyle


class Vilt:
    def __init__(self, device):
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.device = device
        self.processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )
        self.model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )
        self.model.to(self.device)

    def __call__(self, image, question):
        image = image.convert("RGB")
        inputs = self.processor(
            images=image,
            text="how many bears in the image",
            return_tensors="pt",
        ).to(self.device)
        predictions = self.model(**inputs)
        logits = predictions.logits
        idx = logits.argmax(-1).item()
        answer = self.model.config.id2label[idx]
        return answer

    def to(self, device):
        self.model.to(device)


class LLaVA:
    def __init__(self, device):
        self.load_8bit = True if "cuda" in device else False
        self.device = device
        model_name = get_model_name_from_path("liuhaotian/llava-v1.5-7b")
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = load_pretrained_model(
            "liuhaotian/llava-v1.5-7b",
            None,
            model_name,
            self.load_8bit,
            False,
            device=self.device,
        )

        if "llama-2" in model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

    def load_image(self, image_file):
        if image_file.startswith("http://") or image_file.startswith("https://"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def __call__(self, image, question):
        conv = conv_templates[self.conv_mode].copy()
        # roles = conv.roles
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        # Similar operation in model_worker.py
        image_tensor = process_images(
            [image], self.image_processor, {"image_aspect_ratio": "pad"}
        )
        if type(image_tensor) is list:
            image_tensor = [
                image.to(self.device, dtype=torch.float16) for image in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        inp = question
        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + inp
                )
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # streamer = TextStreamer(
        #     self.tokenizer, skip_prompt=True, skip_special_tokens=True
        # )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512,
                # streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        conv.messages[-1][-1] = outputs
        return outputs

    def to(self, device):
        if not self.load_8bit:
            self.model.to(device)


if __name__ == "__main__":
    model = LLaVA("cuda:0")
    output = model(
        "/mnt/afs/user/liuzhaoyang/workspace/graph-of-thought/tests/test_files/FatBear1.jpg",
        "how many bears in this image",
    )
    print(output)
