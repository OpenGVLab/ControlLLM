import os
import os.path as osp
from PIL import Image
import numpy as np
import uuid
import random
import json
from typing import Union
import wget
from PIL import Image
import easyocr
import torch
from torchvision.ops import box_convert
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import hf_hub_download

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict
import groundingdino.datasets.transforms as T

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from .sam_preditor import SamPredictor


CURRENT_DIR = osp.dirname(osp.abspath(__file__))
MODEL_ZOO = os.environ.get("GOT_MODEL_ZOO", "model_zoo")
os.makedirs(MODEL_ZOO, exist_ok=True)

GLOBAL_SEED = 2023


def load_model_hf(model_config_path, repo_id, filename, device="cpu"):
    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


class OCR:
    def __init__(self, device):
        self.device = device
        self.model = easyocr.Reader(["ch_sim", "en"], gpu=device)

    def to(self, device):
        self.model = easyocr.Reader(["ch_sim", "en"], gpu=device)

    def __call__(self, image: Image):
        image = np.array(image)
        result = self.model.readtext(image)
        res_text = self.parse_result(result)
        res_text = json.dumps(res_text, ensure_ascii=False)
        return res_text

    def parse_result(self, result):
        res_text = []
        for item in result:
            # ([[x, y], [x, y], [x, y], [x, y]], text, confidence)
            res_text.append(item[1])
        return res_text


class VisualGrounding:
    def __init__(
        self,
        device: str,
        dtype=torch.float32,
    ):
        self.config_file = osp.join(CURRENT_DIR, "configs/GroundingDINO_SwinT_OGC.py")
        self.ckpt_repo_id = "ShilongLiu/GroundingDINO"
        self.ckpt_filename = "groundingdino_swint_ogc.pth"
        self.model = load_model_hf(
            self.config_file, self.ckpt_repo_id, self.ckpt_filename
        )
        self.box_threshold = 0.8
        self.text_threshold = 0.8
        self.dtype = dtype
        self.device = device
        self.model.to(dtype=dtype)
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def to(self, device):
        self.model.to(device)

    def __call__(self, image: Union[str, Image.Image], query: str):
        if isinstance(image, str):
            if osp.exists(image):
                image = Image.open(image).convert("RGB")
            else:
                return None

        TEXT_PROMPT = query

        image_transformed, _ = self.transform(image, None)

        image_transformed = image_transformed.to(dtype=self.dtype)
        boxes, scores, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=TEXT_PROMPT,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        xyxys = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        coordinates = [
            {
                "score": round(score.item(), 4),
                "label": phrase,
                "box": {
                    "xmin": round(xyxy.tolist()[0], 4),
                    "ymin": round(xyxy.tolist()[1], 4),
                    "xmax": round(xyxy.tolist()[2], 4),
                    "ymax": round(xyxy.tolist()[3], 4),
                },
            }
            for phrase, score, xyxy in zip(phrases, scores, xyxys)
        ]

        return coordinates


class BLIPImageCaptioning:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype
        )
        self.model.to(self.device)

    def __call__(self, image: Image):
        inputs = self.processor(image, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        return captions

    def to(self, device):
        self.model.to(device)


def show_annos(anns):
    # From https://github.com/sail-sg/EditAnything/blob/main/sam2image.py#L91
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    full_img = None

    # for ann in sorted_anns:
    for i in range(len(sorted_anns)):
        ann = anns[i]
        m = ann["segmentation"]
        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
            map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
        map[m != 0] = i + 1
        color_mask = np.random.random((1, 3)).tolist()[0]
        full_img[m != 0] = color_mask
    full_img = full_img * 255
    # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], map.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    full_img = Image.fromarray(np.uint8(full_img))
    return full_img, res


def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        wget.download(url, out=path)
    return path


class SegmentAnything:
    def __init__(self, device):
        self.device = device
        self.model_checkpoint_path = download(
            url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            path=os.path.join(MODEL_ZOO, "sam_vit_h_4b8939.pth"),
        )
        self.sam = sam_model_registry["vit_h"](checkpoint=self.model_checkpoint_path)
        self.predictor = SamPredictor(self.sam)
        self.sam.to(device=device)

    def __call__(self, image: Image):
        image = image.convert("RGB")
        img = np.array(image)
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        annos = mask_generator.generate(img)
        full_img, _ = show_annos(annos)
        return full_img

    def to(self, device):
        self.sam.to(device=device)


class SegmentAnythingStateful(SegmentAnything):
    def __init__(self, device):
        super().__init__(device)
        self.state = {}

    def set_image(self, image: Image):
        image = image.convert("RGB")
        img = np.array(image)
        features = self.predictor.set_image(img)

        image_id = str(uuid.uuid4())
        self.state[image_id] = features
        return image_id

    def segment_by_mask(self, mask: Image, image_id):
        mask = mask.convert("L")
        mask = np.array(mask)
        features = self.state[image_id]
        random.seed(GLOBAL_SEED)
        idxs = np.nonzero(mask)
        num_points = min(max(1, int(len(idxs[0]) * 0.01)), 16)
        sampled_idx = random.sample(range(0, len(idxs[0])), num_points)
        new_mask = []
        for i in range(len(idxs)):
            new_mask.append(idxs[i][sampled_idx])
        points = np.array(new_mask).reshape(2, -1).transpose(1, 0)[:, ::-1]
        labels = np.array([1] * num_points)

        res_masks, scores, _ = self.predictor.predict(
            features=features,
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        output = Image.fromarray(res_masks[np.argmax(scores), :, :])
        return output

    def segment_by_points(self, points: list | tuple, image_id):
        features = self.state[image_id]
        random.seed(GLOBAL_SEED)
        points = np.array(points)
        labels = np.array([1] * len(points))

        res_masks, scores, _ = self.predictor.predict(
            features=features,
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        output = Image.fromarray(res_masks[np.argmax(scores), :, :])
        return output
