import torch
import torch.nn as nn
from transformers import AutoModel, CLIPTokenizer, AutoTokenizer


class Tokenizer:
    def __init__(self, pretrained_model_name_or_path, *inputs, **kwargs):
        if "openai/clip" in pretrained_model_name_or_path:
            tokenizer_builder = CLIPTokenizer
        else:
            tokenizer_builder = AutoTokenizer

        self.tokenizer = tokenizer_builder.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
    
    def __call__(self, texts, **kwargs):
         return self.tokenizer(texts, 
                               return_tensors='pt', 
                               padding='max_length',
                               truncation=True,
                               **kwargs)

class TextClsModel(torch.nn.Module):
    """文本模型

    Args:
    ¦   bert_path (nn.Module): 文本预训练模型路径 '/data/juicefs_hz_cv_v3/11110558/dwork/project/pretrained/simcse-chinese-roberta-wwm-ext'
    """
    def __init__(self, bert_path, class_num=1000, **kwargs):
        super().__init__()
        backbone = AutoModel.from_pretrained(bert_path, output_hidden_states=False, **kwargs)
        hidden_dim = backbone.config.hidden_size
        self.cls_dense = torch.nn.Linear(hidden_dim, class_num)
        self.hidden_dim = hidden_dim
        self.backbone = backbone
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_token = output.last_hidden_state
        text_feature = last_token.mean(dim=1)

        logits = self.cls_dense(text_feature)

        out = {
            'logits': logits,
            'text_feature': text_feature,
            'text_seq_feature': last_token
        }
        return out
    
    def get_dim(self):
        return self.hidden_dim