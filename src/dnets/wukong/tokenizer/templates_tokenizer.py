import os
from .simple_tokenizer import SimpleTokenizer
from typing import Union, List
from ... import data_root

class TemplatesTokenizer:
	def __init__(self, context_len=32, text_templates: Union[str, list] = '{}') -> None:
		self.tokenizer = SimpleTokenizer(context_len=context_len)
		if isinstance(text_templates, str):
			if os.path.isfile(text_templates):
				text_templates = self.load_text_templates(text_templates)	
			else:
				text_templates = [text_templates]
		
		self.text_templates = text_templates
	
	def __len__(self):
		return len(self.text_templates)
	
	def load_text_templates(self, path=os.path.join(data_root, 'wukong', 'prompts.txt')):
		prompts = []
		with open(path) as f:
			for line in f.readlines():
				prompts.append(line.strip())
		return prompts
	
	def __call__(self, words: List[str]):
		texts = []
		for word in words:
			mword = map(lambda x: x.replace("{}", word), self.text_templates)
			texts.extend(mword)
		
		tokens = self.tokenizer.tokenize(texts)
		return tokens

