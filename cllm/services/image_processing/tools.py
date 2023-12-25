from PIL import Image
import numpy as np
import cv2

from transformers import pipeline
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector


class Image2Canny:
    def __init__(self, device='cpu'):
        self.device = device
        self.low_threshold = 100
        self.high_threshold = 200

    def __call__(self, image):
        image = np.array(image)
        canny = cv2.Canny(image, self.low_threshold, self.high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny = Image.fromarray(canny)
        return canny

    def to(self, device):
        pass


class Image2Line:
    def __init__(self, device='cpu'):
        self.device = device
        self.detector = MLSDdetector.from_pretrained('lllyasviel/Annotators')

    def __call__(self, image):
        mlsd = self.detector(image)
        return mlsd

    def to(self, device):
        pass


class Image2Hed:
    def __init__(self, device='cpu'):
        self.device = device
        self.detector = HEDdetector.from_pretrained('lllyasviel/Annotators')

    def __call__(self, image):
        hed = self.detector(image)
        return hed

    def to(self, device):
        pass


class Image2Scribble:
    def __init__(self, device='cpu'):
        self.device = device
        self.detector = HEDdetector.from_pretrained('lllyasviel/Annotators')

    def __call__(self, image):
        scribble = self.detector(image, scribble=True)
        return scribble

    def to(self, device):
        pass


class Image2Pose:
    def __init__(self, device='cpu'):
        self.device = device
        self.detector = OpenposeDetector.from_pretrained('lllyasviel/Annotators')

    def __call__(self, image):
        pose = self.detector(image)
        return pose

    def to(self, device):
        pass


class Image2Depth:
    def __init__(self, device='cpu'):
        self.device = device
        self.depth_estimator = pipeline('depth-estimation')

    def __call__(self, image):
        depth = self.depth_estimator(image)['depth']
        depth = np.array(depth)
        depth = depth[:, :, None]
        depth = np.concatenate([depth, depth, depth], axis=2)
        depth = Image.fromarray(depth)
        return depth

    def to(self, device):
        pass


class Image2Normal:
    def __init__(self, device='cpu'):
        self.device = device
        self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        self.bg_threhold = 0.4

    def __call__(self, image):
        original_size = image.size
        image = self.depth_estimator(image)['predicted_depth'][0]
        image = image.numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)
        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < self.bg_threhold] = 0
        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < self.bg_threhold] = 0
        z = np.ones_like(x) * np.pi * 2.0
        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize(original_size)
        return image

    def to(self, device):
        pass
