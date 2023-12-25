import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from . import imagebind as ib


class Anything2Image:
    def __init__(self, device):
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
        )
        self.device = device
        self.pipe = pipe.to(device)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()

        self.model = ib.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(device)

    def audio2image(self, audio_path):
        embeddings = self.model.forward({
            ib.ModalityType.AUDIO: ib.load_and_transform_audio_data([audio_path], self.device),
        })
        embeddings = embeddings[ib.ModalityType.AUDIO]
        images = self.pipe(image_embeds=embeddings.half(), width=512, height=512).images
        return images[0]

    def thermal2image(self, thermal_path):
        embeddings = self.model.forward({
            ib.ModalityType.THERMAL: ib.load_and_transform_thermal_data([thermal_path], self.device),
        })
        embeddings = embeddings[ib.ModalityType.THERMAL]
        images = self.pipe(image_embeds=embeddings.half(), width=512, height=512).images
        return images[0]

    def audioimage2image(self, image_path, audio_path):
        embeddings = self.model.forward({
            ib.ModalityType.VISION: ib.load_and_transform_vision_data([image_path], self.device),
        }, normalize=False)
        img_embeddings = embeddings[ib.ModalityType.VISION]
        embeddings = self.model.forward({
            ib.ModalityType.AUDIO: ib.load_and_transform_audio_data([audio_path], self.device),
        })
        audio_embeddings = embeddings[ib.ModalityType.AUDIO]
        embeddings = (img_embeddings + audio_embeddings) / 2
        images = self.pipe(image_embeds=embeddings.half(), width=512, height=512).images
        return images[0]

    def audiotext2image(self, audio_path, text):
        embeddings = self.model.forward({
            ib.ModalityType.TEXT: ib.load_and_transform_text([text], self.device),
        }, normalize=False)
        text_embeddings = embeddings[ib.ModalityType.TEXT]

        embeddings = self.model.forward({
            ib.ModalityType.AUDIO: ib.load_and_transform_audio_data([audio_path], self.device),
        })
        audio_embeddings = embeddings[ib.ModalityType.AUDIO]
        embeddings = text_embeddings * 0.5 + audio_embeddings * 0.5
        images = self.pipe(image_embeds=embeddings.half(), width=512, height=512).images
        return images[0]
