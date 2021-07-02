"""
Reference:
    - Randaugment Paper: https://arxiv.org/pdf/1909.13719.pdf
    - RandAugment PyTorch Implementation: https://github.com/ildoonet/pytorch-randaugment/tree/master/RandAugment
"""
import logging
import random
from PIL import Image
from typing import List

from augmentation.augmentations import (
    translateX,
    translateY,
    shearX,
    shearY,
    rotate,
    brightness,
    sharpness,
    equalize,
    color,
    autocontrast,
    posterize,
    solarize,
    cutout,
    contrast,
    identity,
)

logger = logging.getLogger()


# List of augmentations: (augmentation_function, range_min, range_max)
# Ranges taken from https://github.com/google-research/fixmatch
def get_randaug_list() -> List:
    return [
        (translateX, 0, 0.3),
        (translateY, 0, 0.3),
        (shearX, 0, 0.3),
        (shearY, 0, 0.3),
        (rotate, 0, 30),
        (brightness, 0.1, 1.9),
        (sharpness, 0.1, 1.9),
        (equalize, None, None),
        (color, 0.1, 1.9),
        (autocontrast, 0.05, 0.95),
        (posterize, 4, 8),
        (solarize, 0, 255),
        (contrast, 0.1, 1.9),
        (identity, None, None),
    ]


class RandAugment:
    """
    Class that implements RandAugment, the augmentation strategy presented in https://arxiv.org/pdf/1909.13719.pdf
    RandAugment randomly selects and sequentially applies a set of augmentation operations from a list of availble
    operations. The magnitude is computed based on an augmentation-specific predefined range. It can either be fixed or
    randomly sampled at every call.
    """
    def __init__(self, n: int, m: int = 5, max_scale: int = 30, randomized_magnitude: bool = False):
        """
        Initialization of a RandAugment class.

        Parameters
        ----------
        n: int
            Number of augmentation operations, RandAugment samples and sequentially applies at every call.
        m: int (default: 5)
            Strength of augmentation operations applied by RandAugment. The strength should be interpreted relative
            to the defined max_scale, e.g. if m=5 and max_scale=10 the magnitude of augmentation operations will be
            chosen at the half of the defined range:
                augmentation_param = (range_min + (range_max - range_min) * (m / max_scale))
        max_scale: int (default: 30)
            Maximum value of m. The higher max_scale the more granular the augmentation magnitude can be specified.
        randomized_magnitude: bool (default: False)
            Boolean flag indicating whether augmentation magnitudes should be sampled randomly or not. If True, the
            magnitude of every selected augmentation operation at every call is uniformly sampled from the range (0, m).
        """
        assert (
            m <= max_scale
        ), "Invalid configuration. RandAugment magnitude cannot be larger than maximum scale."
        self.N = n
        self.M = m
        self.cutout = cutout
        self.max_scale = max_scale
        self.randomized_magnitude = randomized_magnitude
        self.augmentation_list = get_randaug_list()

    def __call__(self, img: Image):
        """
        Call function of RandAugment class. The method randomly samples N augmentation operations from the list
        of available augmentations and sequentially applies them to the image.

        Parameters
        ----------
        img: Image
            Input image that is augmented using the A
        Returns
        -------
        augmented_img: Image
            Output image after all augmentation operations have been applied.
        """
        augmentations = random.choices(self.augmentation_list, k=self.N)
        for transform, range_min, range_max in augmentations:
            magnitude = self.get_transformation_magnitude(range_min, range_max)
            img = transform(img, magnitude)
        return img

    def get_transformation_magnitude(self, range_min: float, range_max: float):
        """
        Method that returns a transform magnitude based on the given ranges as well as the class variables of
        the RandAugment instance.

        Parameters
        ----------
        range_min: float
            Lower end of the parameter range of the augmentation operation
        range_max: float
            Higher end of the parameter range of the augmentation operation
        Returns
        -------
        magnitude: Optional[float]
            Returns magnitude based on the magnitude, max scale and randomization flag set in the class as well as the
            given range. For augmentation operations that do not have any magnitude parameters, the method returns None.
        """
        if range_min is None or range_max is None:
            return None
        magnitude = random.randint(0, self.M) if self.randomized_magnitude else self.M
        return range_min + (range_max - range_min) / self.max_scale * magnitude
