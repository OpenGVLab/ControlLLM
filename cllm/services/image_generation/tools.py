import os

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, StableDiffusionImageVariationPipeline
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers import PixArtAlphaPipeline

os.environ["CURL_CA_BUNDLE"] = ""

__ALL__ = [
    "Text2Image",
    "Image2Image" "CannyText2Image",
    "LineText2Image",
    "HedText2Image",
    "ScribbleText2Image",
    "PoseText2Image",
    "SegText2Image",
    "DepthText2Image" "NormalText2Image",
]


def resize_800(image):
    w, h = image.size
    if w > h:
        ratio = w * 1.0 / 800
        new_w, new_h = 800, int(h * 1.0 / ratio)
    else:
        ratio = h * 1.0 / 800
        new_w, new_h = int(w * 1.0 / ratio), 800
    image = image.resize((new_w, new_h))
    return image


class Text2Image:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=self.torch_dtype
        )
        self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    def __call__(self, text):
        prompt = text + ", " + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        return image

    def to(self, device):
        self.pipe.to(device)


class PixArtAlpha:
    def __init__(self, device, **kwargs):
        self.device = device
        self.device = device
        self.dtype = kwargs.get("dtype", torch.float16)
        self.pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=self.dtype
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.to(self.device)

    def __call__(self, text):
        return self.pipe(text).images[0]

    def to(self, device=None):
        self.pipe.to(device)
        return self


class Image2Image:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            revision="v2.0",
            torch_dtype=self.torch_dtype,
        )
        self.pipe.to(device)

    def __call__(self, image):
        image = self.pipe(image).images[0]
        return image

    def to(self, device):
        self.pipe.to(device)
        return self


class CannyText2Image:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-canny",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    def __call__(self, image: Image, text):
        w, h = image.size
        image = resize_800(image)
        prompt = f"{text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image = image.resize((w, h))
        return image

    def to(self, device):
        self.pipe.to(device)
        return self


class LineText2Image:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-mlsd", torch_dtype=self.torch_dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    def __call__(self, image, text):
        w, h = image.size
        image = resize_800(image)
        prompt = f"{text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image = image.resize((w, h))
        return image

    def to(self, device):
        self.pipe.to(device)
        return self


class HedText2Image:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-hed", torch_dtype=self.torch_dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    def __call__(self, image, text):
        w, h = image.size
        image = resize_800(image)
        prompt = f"{text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image = image.resize((w, h))
        return image

    def to(self, device):
        self.pipe.to(device)
        return self


class ScribbleText2Image:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-scribble",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    def __call__(self, image, text):
        w, h = image.size
        image = resize_800(image)
        prompt = f"{text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image = image.resize((w, h))
        return image

    def to(self, device):
        self.pipe.to(device)
        return self


class PoseText2Image:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-openpose",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.num_inference_steps = 20
        self.unconditional_guidance_scale = 9.0
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,"
            " fewer digits, cropped, worst quality, low quality"
        )

    def __call__(self, image, text):
        w, h = image.size
        image = resize_800(image)
        prompt = f"{text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image = image.resize((w, h))
        return image

    def to(self, device):
        self.pipe.to(device)
        return self


class SegText2Image:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-seg", torch_dtype=self.torch_dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,"
            " fewer digits, cropped, worst quality, low quality"
        )

    def __call__(self, image, text):
        w, h = image.size
        image = resize_800(image)
        prompt = f"{text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image = image.resize((w, h))
        return image

    def to(self, device):
        self.pipe.to(device)
        return self


class DepthText2Image:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-depth",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,"
            " fewer digits, cropped, worst quality, low quality"
        )

    def __call__(self, image, text):
        w, h = image.size
        image = resize_800(image)
        prompt = f"{text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image = image.resize((w, h))
        return image

    def to(self, device):
        self.pipe.to(device)
        return self


class NormalText2Image:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-normal",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,"
            " fewer digits, cropped, worst quality, low quality"
        )

    def __call__(self, image, text):
        w, h = image.size
        image = resize_800(image)
        prompt = f"{text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image = image.resize((w, h))
        return image

    def to(self, device):
        self.pipe.to(device)
        return self
