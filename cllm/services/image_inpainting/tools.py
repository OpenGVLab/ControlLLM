import os

import cv2
import numpy as np
import torch
import wget
from omegaconf import OmegaConf
from PIL import Image

from .ldm_inpainting.ldm.models.diffusion.ddim import DDIMSampler
from .ldm_inpainting.ldm.util import instantiate_from_config

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def cal_dilate_factor(mask):
    area = mask[mask != 0].sum()
    edge = cv2.Canny(mask, 30, 226)
    perimeter = edge.sum()
    ratio = 0
    if perimeter > 0:
        ratio = int(area * 0.55 / perimeter)
    if ratio % 2 == 0:
        ratio += 1
    return ratio


def dilate_mask(mask, dilate_factor=9):
    # dilate mask
    mask = mask.astype(np.uint8)
    dilated_mask = cv2.dilate(mask, np.ones((dilate_factor, dilate_factor), np.uint8), iterations=1)

    return dilated_mask


def make_batch(image, mask, device):
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1 - mask) * image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0
    return batch


class LDMInpainting:
    def __init__(self, device):
        self.model_checkpoint_path = 'model_zoo/ldm_inpainting_big.ckpt'
        config = os.path.join(CURRENT_DIR, 'ldm_inpainting/config.yaml')
        self.ddim_steps = 50
        self.device = device
        config = OmegaConf.load(config)
        model = instantiate_from_config(config.model)
        self.download_parameters()
        model.load_state_dict(torch.load(self.model_checkpoint_path)["state_dict"], strict=False)
        self.model = model.to(device=device)
        self.sampler = DDIMSampler(model)

    def download_parameters(self):
        url = 'https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1'
        if not os.path.exists(self.model_checkpoint_path):
            wget.download(url, out=self.model_checkpoint_path)

    @torch.no_grad()
    def __call__(self, image, mask):
        mask = mask.convert('L')
        w, h = image.size
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))
        image = np.array(image)
        mask = np.array(mask)
        dilate_factor = cal_dilate_factor(mask.astype(np.uint8))
        mask = dilate_mask(mask, dilate_factor)

        with self.model.ema_scope():
            batch = make_batch(image, mask, device=self.device)
            # encode masked image and concat downsampled mask
            c = self.model.cond_stage_model.encode(batch["masked_image"])
            cc = torch.nn.functional.interpolate(batch["mask"],
                                                 size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)

            shape = (c.shape[1] - 1,) + c.shape[2:]
            samples_ddim, _ = self.sampler.sample(S=self.ddim_steps,
                                                  conditioning=c,
                                                  batch_size=c.shape[0],
                                                  shape=shape,
                                                  verbose=False)
            x_samples_ddim = self.model.decode_first_stage(samples_ddim)

            image = torch.clamp((batch["image"] + 1.0) / 2.0,
                                min=0.0, max=1.0)
            mask = torch.clamp((batch["mask"] + 1.0) / 2.0,
                               min=0.0, max=1.0)
            predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                          min=0.0, max=1.0)

            inpainted = (1 - mask) * image + mask * predicted_image
            inpainted = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255

        inpainted = inpainted.astype(np.uint8)
        new_img = Image.fromarray(inpainted)
        new_img = new_img.resize((w, h))
        return new_img

    def to(self, device):
        self.model.to(device)
