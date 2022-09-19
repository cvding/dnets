import os
import torch
from torchvision import transforms
from .resnet import resnest101
from ..utils import _to_pil, load_wgts
from .. import data_root
from ..base import Inference


class GeneralRecognition(Inference):
    def __init__(self, conf):
        super().__init__(models=[resnest101], conf=conf)

        task = self.conf.get_conf('inference', 'use')
        self.model = self.conf.build('resnest101', task) 
        aconf = self.conf.get_conf('inference', task)
        ckpt = torch.load(aconf['model_path'], map_location='cpu')
        load_wgts(self.model, ckpt)

        self.topk = aconf['topk']
        self.device = aconf['device'] 
        self.model.to(device=self.device)
        self.model.eval()

        self.label_mapping = []
        label_mapping_path = aconf['label_mapping_path']
        if not os.path.exists(label_mapping_path):
            label_mapping_path=os.path.join(data_root, 'classify', aconf['label_mapping_path'])

        with open(label_mapping_path, 'r') as f:
            for line in f.readlines():
                texts = line.strip().split('\t')
                self.label_mapping.append(texts[1])
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), normalize
        ])
        
    def preprocess(self, input):
        img_buf = input['img']
        img = _to_pil(img_buf)
        img = self.test_transforms(img)
        result = {'img': torch.unsqueeze(img, dim=0)}
        return result
    
    def postprocess(self, inputs):
        out = torch.softmax(inputs['outputs'], dim=1)
        out_val, out_idx = torch.topk(out, k=self.topk)
        inputs = {
            "score": out_val[0].cpu().numpy(),
            "label": [self.label_mapping[idx.item()] for idx in out_idx[0]]
        }
        return inputs
    
    def forward(self, inputs):
        img = inputs['img']
        img = img.to(self.device)
        outputs = self.model(img)
        return {'outputs': outputs}
    
    def bpreprocess(self, inputs):
        imgs = [_to_pil(img) for img in inputs]
        img_tensor = [self.test_transforms(img) for img in imgs]
        imgs_tensor = torch.stack(img_tensor, dim=0)

        return {"img": imgs_tensor}
    
    def bpostprocess(self, inputs):
        out = torch.softmax(inputs['outputs'], dim=1)
        out_val, out_idx = torch.topk(out, k=self.topk)

        out_name = []
        for i in range(out.shape[0]):
            top_name = [self.label_mapping[idx.item()] for idx in out_idx[i]]
            out_name.append(top_name)
        
        return {"score": out_val.cpu().numpy(), "label": out_name}




