import os
import torch
from .. import data_root
from ..base import Inference, Conf
from typing import Dict, Any
from ..utils import load_wgts, _to_pil
from .clip import CLIP
from torchvision.transforms import Normalize, Compose, Resize, ToTensor, InterpolationMode
from tokenizers import BertWordPieceTokenizer


class ClipEmbedding(Inference):
    def __init__(self, conf: Conf) -> None:
        super().__init__([CLIP], conf)

        gconf = self.conf.get_conf('inference', 'common')

        self.context_len = gconf['context_len']
        self.device = gconf['device']
        self.model = self.conf.build('CLIP', gconf['use'])

        ckpt = torch.load(gconf['model_path'], map_location='cpu')
        load_wgts(self.model, ckpt, True, True)
        self.model.to(self.device)
        self.model.eval()
    
        img_size = gconf['input_resolution']
        norm_op = Normalize((0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711))
        self.image_transform = Compose([
            Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            ToTensor(), norm_op
        ])

        # text tokenizer
        vocab_path = os.path.join(data_root, 'wukong', 'vocab.txt')
        self.text_tokenizer = BertWordPieceTokenizer(
            vocab_path, lowercase=False)
        self.text_tokenizer.enable_truncation(max_length=gconf['context_len'])

    @torch.no_grad() 
    def tokenize_text(self, text_str):
        tokens = self.text_tokenizer.encode(text_str)
        max_tokens = self.context_len - 2
        text_ids_tensor = torch.zeros((1, max_tokens)).long()
        text_mask_tensor = torch.zeros((1, max_tokens))

        text_ids, text_mask = tokens.ids, tokens.attention_mask
        text_ids_tensor[0, 0:len(text_ids)] = torch.tensor(text_ids)
        text_mask_tensor[0, 0:len(text_mask)] = torch.tensor(text_mask)

        return text_ids_tensor, text_mask_tensor

    @torch.no_grad() 
    def preprocess(self, input):
        img = _to_pil(input['img'])
        img = self.image_transform(img)
        img = torch.unsqueeze(img, dim=0)

        img = img.to(self.device)
        input['img'] = img

        text_ids_tensor, text_mask_tensor = self.tokenize_text(input['txt'])
        text_ids_tensor = text_ids_tensor.to(self.device)
        text_mask_tensor = text_mask_tensor.to(self.device)

        input['txt'] = (text_ids_tensor, text_mask_tensor)

        return input
    
    @torch.no_grad()
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        img_embedding, txt_embedding, _ = self.model(input['img'], input['txt'])
        return {"img_embedding": img_embedding, "txt_embedding": txt_embedding}