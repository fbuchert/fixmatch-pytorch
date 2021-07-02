"""
Reference:
    - PIL: https://pillow.readthedocs.io/en/stable/
"""
import random
from typing import List
from PIL import Image, ImageEnhance, ImageOps, ImageDraw


import numpy as np
from torchvision import transforms
from datasets.config import *


def get_weak_transforms() -> List:
    return [
        (translateX, 0, 0.125),
        (translateY, 0, 0.125),
        (random_horizontal_flip, None, None),
    ]


def identity(img: Image, _) -> Image:
    return img


def translateX(img: Image, mag: float):
    mag = mag * img.size[0]
    return img.transform(
        img.size, Image.AFFINE, (1, 0, mag * random.choice([-1, 1]), 0, 1, 0)
    )


def translateY(img: Image, mag: float):
    mag = mag * img.size[1]
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, 0, 1, mag * random.choice([-1, 1]))
    )


def shearX(img: Image, mag: float):
    return img.transform(
        img.size, Image.AFFINE, (1, mag * random.choice([-1, 1]), 0, 0, 1, 0)
    )


def shearY(img: Image, mag: float) -> Image:
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, mag * random.choice([-1, 1]), 1, 0)
    )


def rotate(img: Image, mag: float) -> Image:
    return img.rotate(mag * random.choice([-1, 1]))


def brightness(img: Image, mag: float) -> Image:
    assert 0.0 <= mag <= 2.0
    return ImageEnhance.Brightness(img).enhance(mag)


def sharpness(img: Image, mag: float) -> Image:
    assert 0.0 <= mag <= 2.0
    return ImageEnhance.Sharpness(img).enhance(mag)


def equalize(img: Image, _) -> Image:
    return ImageOps.equalize(img)


def color(img: Image, mag: float) -> Image:
    assert 0.0 <= mag <= 2.0
    return ImageEnhance.Color(img).enhance(mag)


def autocontrast(img: Image, mag: float) -> Image:
    assert 0.0 <= mag <= 1.0
    return ImageOps.autocontrast(img, mag)


def contrast(img: Image, mag: float) -> Image:
    assert 0.0 <= mag <= 2.0
    return ImageEnhance.Contrast(img).enhance(mag)


def posterize(img: Image, mag: float) -> Image:
    assert 1 <= int(mag) <= 8
    return ImageOps.posterize(img, int(mag))


def solarize(img: Image, mag: float) -> Image:
    assert 0 <= int(mag) < 256
    return ImageOps.solarize(img, int(mag))


def random_horizontal_flip(img: Image, _, p: float = 0.5):
    if random.uniform(0, 1) <= p:
        img = ImageOps.mirror(img)
    return img


def cutout(img: Image, mag: float):
    assert 0.0 <= mag <= 1
    mag = mag * img.size[0]
    return cutoutAbs(img, mag)


def cutoutAbs(img: Image, mag: float = 0.5):
    if mag < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - mag / 2.0))
    y0 = int(max(0, y0 - mag / 2.0))
    x1 = min(w, x0 + mag)
    y1 = min(h, y0 + mag)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


def get_normalizer(dataset: str):
    """
    Method that returns normalization transform for commonly used image datasets.

    Parameters
    ----------
    dataset: str
        Name of dataset. The dataset has to be supported by NORMALIZATION_VARIABLES defined in datasets.config
    Returns
    -------
    normalization_transform: transforms.Compose
        Returns composition, i.e. class that sequentially applies, transformation of PIL image to Tensor and
        the normalization operation (each channel is normalized to 0 mean and 1 standard deviation).
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**NORMALIZATION_VARIABLES[dataset])
    ])


def get_weak_augmentation(img_size: int = 32, padding: int = 4, padding_mode: str = "reflect"):
    """
    Get weak torch augmentation strategy comprised of a random crop and random horizontal flips.
    This is the default augmentation strategy used in MixMatch, which randomly crops 12.5% of input images.

    Parameters
    ----------
    img_size:
        Desired size of output images, which is used as input to the RandomCrop function. In MixMatch size of input
        and output images of the weak augmentation strategy should be equal.
    padding: int
        Number of padding pixels at added to the border of the image before cropping.
    padding_mode: str (default: "reflect")
        Type of padding used before cropping the image. Options are `constant`, `edge`, `reflect`, `symmetric`.
        Read more at https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomCrop.
    Returns
    -------
    transform: transforms.Compose
        Returns composition, i.e. class that sequentially applies given transformation, of RandomCrop and Random
        Horizontal flip.
    """
    return transforms.Compose(
        [
            transforms.RandomCrop(img_size, padding=padding, padding_mode=padding_mode),
            transforms.RandomHorizontalFlip(),
        ]
    )
