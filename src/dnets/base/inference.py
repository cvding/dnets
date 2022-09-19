import torch
from bindconf import BindConf
from typing import Any, Dict, List
from .types import Input, Conf

class Inference:
    def __init__(self, models:list, conf:Conf) -> None:
        # 构建默认的配置
        self.conf = BindConf(models, conf)
    
    def preprocess(self, input: Input) -> Dict[str, Any]:
        return input

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
    
    def forward(self, inputs):
        return inputs
    
    @torch.no_grad()
    def __call__(self, input) -> Dict[str, Any]:
        inputs = self.preprocess(input)
        inputs = self.forward(inputs)
        inputs = self.postprocess(inputs)

        return inputs
    
    def bpreprocess(self, inputs: List[Input]) -> Dict[str, Any]:
        raise NotImplementedError
    
    def bpostprocess(self, inputs: Dict[str, Any]):
        raise NotImplementedError

    @torch.no_grad()
    def batch(self, inputs: List[Input]) -> Dict[str, Any]:
        inputs = self.bpreprocess(inputs)
        inputs = self.forward(inputs)
        inputs = self.bpostprocess(inputs)

        return inputs
