import os
import torch
import torch.nn as nn
from torchvision import transforms
from .resnet import resnest101
from ..utils import _to_pil, load_wgts
from .. import data_root


class Animal(object):
    def __init__(self, model_path, device='cuda:0', **kwargs):
        num_classes = 8288
        self.model = resnest101(num_classes=num_classes)
        ckpt = torch.load(model_path, map_location='cpu')
        load_wgts(self.model, ckpt)
        
        self.model.to(device=device)
        self.model.eval()

        self.label_mapping = []
        label_mapping_path=os.path.join(data_root, 'animal/label_mapping.txt')
        with open(label_mapping_path, 'r') as f:
            for line in f.readlines():
                texts = line.strip().split('\t')
                self.label_mapping.append(texts[1])
        
    def preprocess(self, input):
        img = _to_pil(input)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), normalize
        ])
        img = test_transforms(img)
        result = {'img': img}
        return result
    
    def postprocess(self, inputs):
        score = torch.max(inputs['outputs'])
        inputs = {
            "score": score.item(),
            "label": self.label_mapping[inputs['outputs'].argmax()]
        }
        return inputs

    @torch.no_grad()
    def inference(self, input):
        self.model.eval()
        img = input['img']
        input_img = torch.unsqueeze(img, 0)
        outputs = self.model(input_img)
        return {'outputs': outputs}
    
    def forward(self, input):
        return self.model(input)

    

    


        