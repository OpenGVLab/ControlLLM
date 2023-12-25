from transformers import (
    pipeline,
    AutoModel,
    AutoProcessor,
    MusicgenForConditionalGeneration,
)
from PIL import Image
import torch
import scipy
import io
import numpy as np


'''
class Text2Speech:
    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = AutoModel.from_pretrained("suno/bark-small")
        # self.model.to(self.device)

    def __call__(self, text):
        inputs = self.processor(
            text = [text],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        audio_values = self.model.generate(**inputs, do_sample=True)
        
        # TODO
        save_path = 'resources/test.wav'
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(save_path, rate=sampling_rate, data=audio_values[0, 0].numpy())
        return save_path

    def to(self, device):
        self.model.to(device)
'''


class Text2Music:
    def __init__(self, device):
        self.device = device
        self.dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(
            "facebook/musicgen-small"
        )
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small", torch_dtype=self.dtype
        )
        self.model.to(device=self.device)

    def __call__(self, text: str):
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        audio_values = self.model.generate(**inputs, max_new_tokens=512)

        # TODO
        stream = io.BytesIO()
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(
            stream,
            rate=sampling_rate,
            data=audio_values[0, 0].cpu().numpy().astype(np.float32),
        )
        stream.seek(0)
        return stream

    def to(self, device):
        self.device = device
        self.model.to(device)


if __name__ == "__main__":
    model = Text2Music('auto')
    print(
        model(
            "An 80s driving pop song with heavy drums and synth pads in the background"
        )
    )
