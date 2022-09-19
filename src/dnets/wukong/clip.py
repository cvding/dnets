import os
import json
import torch
from ..base import Inference, Conf
from typing import Dict, Any
from .wukong import Wukong
import PIL
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from ..utils import _to_pil
from .tokenizer import SimpleTokenizer, TemplatesTokenizer
from .. import data_root
from .utils import token_wise_similarity


class WuKongEmbedding(Inference):
	def __init__(self, conf: Conf) -> None:
		super().__init__([Wukong], conf)
		mean = (0.48145466, 0.4578275, 0.40821073)
		std = (0.26862954, 0.26130258, 0.27577711)
		gconf = self.conf.get_conf('Wukong', 'common')

		self.device = gconf['device']
		self.model = Wukong(self.conf.get_conf(None, None))
		self.model.to(self.device)
		self.model.eval()
		self.img_transform = Compose([
			Resize(gconf['input_resolution'], interpolation=PIL.Image.BICUBIC),
        	CenterCrop(gconf['input_resolution']),
        	lambda img: img.convert("RGB"),
        	ToTensor(),
        	Normalize(mean, std),
		])
		self.tokenizer = SimpleTokenizer(context_len=gconf['context_len'])

	def preprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
		# preprocess image
		img = _to_pil(input['img'])
		img = self.img_transform(img)
		img = img.unsqueeze(dim=0)
		img = img.to(self.device)	
		input['img'] = img

		# preprocess text
		txt = self.tokenizer.tokenize(input['txt'])
		input['txt'] = txt.to(self.device)
		return input
	
	def forward(self, inputs):
		image_features = self.model.encode_image(inputs['img'])
		image_features = self.model.process_img_features(image_features)

		text_feature = self.model.encode_text(inputs['txt'])
		text_feature = self.model.process_text_features(text_feature, inputs['txt'])
		return {"image_embedding": image_features, "text_embedding": text_feature}


class WuKongClassify(WuKongEmbedding):
	def __init__(self, conf: Conf) -> None:
		super().__init__(conf)

		cls_conf = self.conf.get_conf('Wukong', 'classify')
		name_dict_path = cls_conf['name_dict_path'] or os.path.join(data_root, 'wukong', 'imagenet.names.json')
		prompts_path = cls_conf['prompts_path'] if cls_conf['prompts_path'] is not None else '{}'
		self.topk = cls_conf['topk']

		gconf = self.conf.get_conf('Wukong', 'common')
		temp_tokenizer = TemplatesTokenizer(gconf['context_len'], text_templates=prompts_path)

		with open(name_dict_path, 'r') as f:
			jdict = json.load(f)
			class_names = [name for name in jdict.values()]
			name_dict = dict([(idx, name) for idx, name in enumerate(jdict.values())])
		
		text_tokens = temp_tokenizer(class_names)
		self.num_prompts = len(temp_tokenizer)
		self.name_dict = name_dict
		self.text_features = self.__get_text_features(text_tokens.to(self.device))
		
	
	@torch.no_grad()
	def __get_text_features(self, text_tokens):
		"""获取文本特征

		Args:
			model (nn.Module): 悟空模型
			dataset (_type_): _description_
			prompts (_type_): _description_
			num_classes (类别数): 列表标签总数

		Returns:
			_type_: _description_
		"""
		texts = text_tokens 
		text_batch_size = 1024
		text_features = []
		num_classes = len(self.name_dict.keys())

		for i in range((len(texts) // text_batch_size) + 1):
			text = texts[i * text_batch_size: (i + 1) * text_batch_size]
			if len(text):
				text_features_ = self.model.encode_text(text)
				text_features_ = self.model.process_text_features(text_features_, text)
				text_features.append(text_features_)
		text_features = torch.cat(text_features)
		# prompt ensemble
		if self.num_prompts > 1:
			text_features = text_features.reshape(
				num_classes, self.num_prompts, *text_features.shape[1:]
			)
			if not self.model.is_token_wise:
				text_features = text_features.mean(1)
				text_features /= text_features.norm(dim=-1, keepdim=True)
		return text_features
	
	@staticmethod
	@torch.no_grad()
	def __get_logits(image_features, text_features, is_token_wise=False):
		logits = (
			token_wise_similarity(image_features, text_features).softmax(dim=-1)
			if is_token_wise
			else (image_features @ text_features.T).softmax(dim=-1)
		)
		return logits

	def preprocess(self, input):
		img = _to_pil(input['img'])
		img = self.img_transform(img)
		img = img.unsqueeze(dim=0)
		img = img.to(self.device)	
		input['img'] = img

		return input

	def forward(self, inputs):
		image_features = self.model.encode_image(inputs['img'])
		image_features = self.model.process_img_features(image_features)
		
		return {'img_embedding': image_features}
		
	
	def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
		image_features = inputs['text_embedding']
		logits = self.__get_logits(image_features, self.text_features, self.model.is_token_wise)
		out_val, out_idx = logits.topk(self.topk)

		outputs = {
			"score": out_val[0].cpu().numpy(),
			"label": [self.name_dict[idx.item()] for idx in out_idx[0]]
		}

		return outputs
		
		


