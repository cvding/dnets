import torch
from typing import Union
from PIL import Image
import numpy as np

Tensor = torch.Tensor
Input = Union[str, tuple, 'Image.Image', 'np.ndarray']
Conf = Union[str, dict]
ImgBuf = Union[str, bytes, 'Image.Image', 'np.ndarray']

