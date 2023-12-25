import tempfile
from transformers import (
    pipeline,
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
)
from modelscope.pipelines import pipeline as mspipeline
from modelscope.outputs import OutputKeys
from modelscope.preprocessors import Preprocessor
from modelscope.models.base import Model
from modelscope.utils.constant import Invoke
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from moviepy.editor import ImageSequenceClip

from PIL import Image
import av
import os.path as osp
import numpy as np
import cv2
from pathlib import Path
import moviepy.editor as mpe
import torch
import uuid
import os
from einops import rearrange

RESOURCE_ROOT = os.environ.get("RESOURCE_ROOT", "./server_resources")


class Image2Video:
    def __init__(self, device):
        self.device = device
        model_path = "damo/Image-to-Video"
        self.preprocessor = Preprocessor.from_pretrained(model_path)
        self.model = Model.from_pretrained(
            model_path,
            invoked_by=Invoke.PIPELINE,
            device_map="auto",
        )
        self.model.to(self.device)

    def tensor2vid(self, video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
        std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)

        video = video.mul_(std).add_(mean)
        video.clamp_(0, 1)
        video = video * 255.0

        images = rearrange(video, "b c f h w -> b f h w c")[0]
        images = [(img.numpy()).astype("uint8") for img in images]

        return images

    def __call__(self, image: Image):
        image = image.convert("RGB")
        vit_frame = self.model.vid_trans(image)
        vit_frame = vit_frame.unsqueeze(0)
        vit_frame = vit_frame.to(self.device)
        inputs = {"vit_frame": vit_frame}
        video_tensors = self.model(inputs)
        frames = self.tensor2vid(video_tensors, self.model.cfg.mean, self.model.cfg.std)
        fps = 4  # Set the desired frame rate (frames per second)
        clip = ImageSequenceClip(frames, fps=fps)
        output_video = osp.join(RESOURCE_ROOT, f"{str(uuid.uuid4())[:6]}.mp4")
        clip.write_videofile(output_video)
        output = open(output_video, "rb").read()
        os.remove(output_video)
        return output

    def to(self, device):
        self.model.to(device=device)
        return self


class TimeSformerGPT2VideoCaptioning:
    def __init__(self, device):
        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "Neleac/timesformer-gpt2-video-captioning"
        ).to(device)

    def __call__(self, video):
        container = av.open(video)
        # extract evenly spaced frames from video
        seg_len = container.streams.video[0].frames
        clip_len = self.model.config.encoder.num_frames
        indices = set(
            np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64)
        )
        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))

        # generate caption
        gen_kwargs = {
            "min_length": 20,
            "max_length": 500,
            "num_beams": 8,
        }
        pixel_values = self.image_processor(
            frames, return_tensors="pt"
        ).pixel_values.to(self.device)
        tokens = self.model.generate(pixel_values, **gen_kwargs)
        caption = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        return caption

    def to(self, device):
        self.model.to(device)


class ImageAudio2Video:
    def __init__(self, device):
        self.device = device

    def __call__(self, image_path, audio_path):
        # Create an empty video object
        # image = cv2.imread(image_path)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)[:, :, ::-1]
        height, width, _ = image.shape
        video_save_path = f"{osp.splitext(osp.basename(image_path))[0]}.mp4"
        video = cv2.VideoWriter(
            video_save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            1,
            (width, height),
        )  # Set the video format, frame rate and resolution

        # Loop through the list of frames, write each frame into the video object
        for frame in [
            image,
        ] * 15:
            # image = cv2.imread(frames_path + frame) # Read the frame image
            video.write(frame)  # Write into the video object

        # Release the video object
        video.release()

        # Read the audio object
        audio = mpe.AudioFileClip(audio_path)

        # Read the video object
        video = mpe.VideoFileClip(video_save_path)

        # Merge the audio and video
        final_video = video.set_audio(audio)

        # Export the final video file
        final_video.write_videofile(video_save_path)
        return video_save_path

    def to(self, device):
        return self


class Video2WebPage:
    def __init__(self, device):
        self.device = device
        self.template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{{title}}</title>
            <style>
                body {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                }

                h1 {
                    text-align: center;
                }

                video {
                    max-width: 100%;
                }

                div {
                    text-align: justify;
                    margin: 10px 10px;
                }

                h2 {
                    text-align: center;
                }

            </style>
        </head>
        <body>
            <h1>{{title}}</h1>

            <!-- Play video -->
            <video controls width="800">
                <source src="{{video_path}}" type="video/mp4">
            </video>

            <!-- Tags -->
            <div style="font-size:20px;">
                <strong>Tags: </strong>
                {{tags}}
                <!-- add tags -->
            </div>

            <!-- Video description -->
            <div>
                <h2>Video description</h2>
                <p style="font-size:20px;">{{description}}</p>
            </div>
        </body>
        </html>
        """

    def __call__(self, video_path: Path, title, tags, description):
        page = self.template.replace("{{video_path}}", video_path)
        page = page.replace("{{title}}", title)
        tags_str = ""
        for i, tag in enumerate(tags):
            if i < len(tags) - 1:
                tags_str += f"<span>{tag};</span> "
            else:
                tags_str += f"<span>{tag}.</span>"

        page = page.replace("{{tags}}", tags_str)
        page = page.replace("{{description}}", description)
        return page

    def to(self, device):
        return self


class DubVideo:
    def __init__(self, device):
        self.device = device

    def __call__(self, video_path: Path, audio_path: Path):
        video = mpe.VideoFileClip(video_path)

        # read audio file
        audio = mpe.AudioFileClip(audio_path)

        # set audio for video
        new_video = video.set_audio(audio)

        # export the video file
        save_path = osp.join(RESOURCE_ROOT, f"{str(uuid.uuid4())[:6]}.mp4")
        new_video.write_videofile(save_path)
        return save_path

    def to(self, device):
        return self


class Text2Video:
    def __init__(self, device):
        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe = self.pipe.to(self.device)

    def __call__(self, prompt: str):
        video_frames = self.pipe(prompt, num_inference_steps=25, num_frames=100).frames
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w, c = video_frames[0].shape
        video_writer = cv2.VideoWriter(
            output_video_path, fourcc, fps=10, frameSize=(w, h)
        )
        for i in range(len(video_frames)):
            img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
            video_writer.write(img)
        video_writer.release()

        video = mpe.VideoFileClip(output_video_path)

        # Export the final video file
        video.write_videofile(output_video_path)

        return open(output_video_path, "rb").read()

    def to(self, device):
        self.pipe = self.pipe.to(device)
        return self


if __name__ == "__main__":
    # model = TimeSformerGPT2VideoCaptioning('cuda:0')
    # print(model("./tests/test_data/test.mp4"))
    model = Image2Video("cuda:0")
    output = model(Image.open("./server_resources/a07d14_image.png"))
    # model = Text2Video("cuda:0")
    # output = model("Spiderman is surfing")
    print(type(output))
    # with open("./tests/test_data/output.mp4", "wb") as file_object:
    #     file_object.write(output)
