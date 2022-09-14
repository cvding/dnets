import torch
from typing import Any, Dict, Union
from PIL import Image
import numpy as np

Tensor = Union['torch.Tensor']
Input = Union[str, tuple, 'Image.Image', 'np.ndarray']

class Inference:
    def __init__(self, conf) -> None:
        self.model = None
    
    def preprocess(self, input: Input) -> Dict[str, Any]:
        output = {"img": None}

        return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
    
    @torch.no_grad()
    def __call__(self, input) -> Dict[str, Any]:
        inputs = self.preprocess(input)
        inputs = self.model(inputs)
        inputs = self.postprocess(inputs)

        return inputs
        


    @torch.no_grad()
    def batch(self, *args: Any, **kwds: Any) -> Any:
        pass