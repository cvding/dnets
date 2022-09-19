import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout
from .model import TransFG, Encoder
from copy import deepcopy as dcopy


class TokenEmbedding(nn.Module):
    def __init__(self, num_patches, hidden_size, dropout_rate):
        super(TokenEmbedding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches+1, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class TokenLearner(nn.Module):
    """Token学习

        Args:
            num_patchs (int): patch的数量
            patch_dim (int): patch的维度
            mlp_dim (int): 全连接层的维度
            num_heads (int): 多头注意数量
            num_layers (int): attention模块的数量
            mlp_dropout (float, optional): 全连接层的dropout. Defaults to 0.1.
            attention_dropout (float, optional): 注意力模块的dropout. Defaults to 0.0.
            num_classes (int, optional): 输入类别数, Defaults to None. 
        """
    def __init__(self, num_patchs, patch_dim, mlp_dim, num_heads, num_layers, mlp_dropout=0.1, attention_dropout=0.0, num_classes=None):
        
        super().__init__()
        params = {
            "num_layers": num_layers,
            "hidden_size": patch_dim,
            "mlp_dim": mlp_dim,
            "num_heads": num_heads,
            "mlp_dropout_rate": mlp_dropout,
            "attention_dropout_rate": attention_dropout
        }
        self.embeddings = TokenEmbedding(num_patchs, hidden_size=patch_dim, dropout_rate=mlp_dropout)
        self.encoder = Encoder(**params)

        if isinstance(num_classes, int):
            self.cls_layer = nn.Linear(patch_dim, num_classes)
        else:
            self.cls_layer = nn.Identity()
    
    def forward(self, x, mask=None):
        # x: [B, T, C] return: [B, self.num_heads, C]
        x = self.embeddings(x)
        tokens = self.encoder(x, mask)
        token = tokens['tokens']
        logits = self.cls_layer(token[:, 0])
        return {"logits": logits, "connect": tokens['connect'], "select": tokens['select']}


class VideoTransFG(nn.Module):
    def __init__(self, transfg_config:dict, video_config:dict, img_size, num_classes, seq_len):
        """创建视频细粒度分类模型

        Args:
            transfg_config (dict): TransFG模型参数
            video_config (dict): 视频Encoder参数,通过video_config()方法获取默认参数
            img_size (tuple): 视频帧尺寸大小
            num_classes (int): 分类的数量
            seq_len (int): 视频帧长度
        """
        super(VideoTransFG, self).__init__()
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = (img_size[0], img_size[1])
        self.backbone = TransFG(img_size=img_size, num_classes=num_classes, **transfg_config)

        hidden_size = transfg_config['hidden_size']
        num_heads = transfg_config['num_heads']
        dropout_rate = transfg_config['embedding_dropout_rate']
        self.seq_len = seq_len
        self.num_heads = num_heads
        num_patchs = num_heads * seq_len 
        self.embeddings = TokenEmbedding(num_patches=num_patchs, hidden_size=hidden_size, dropout_rate=dropout_rate)
        transformer = dcopy(video_config) 
        transformer['hidden_size'] = hidden_size
        self.encoder = Encoder(**transformer)
        self.video_layer = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x.shape [B, T*3, H, W]
        B, T = x.shape[:2]
        T = T // 3
        x = x.view(-1, 3, *self.img_size)
        bone_token_dict = self.backbone(x)
        tokens = bone_token_dict["tokens"]
        tpad = (self.seq_len - T) * self.num_heads
        tokens = tokens[:,1:,:].reshape(B, self.num_heads * T, -1)
        Ts = tokens.size(1)
        tokens = F.pad(tokens, (0, 0, 0, tpad))
        Td = tokens.size(1)
        mask = torch.zeros((B, self.num_heads, Td+1, Td+1), dtype=torch.uint8, device=x.device)
        mask[:, :, :Ts+1, :Ts+1] = 1
        embedding = self.embeddings(tokens)
        seq_token_dict = self.encoder(embedding, mask)
        tokens = seq_token_dict['tokens']
        logits = self.video_layer(tokens[:, 0])

        top_sel = {"connect": bone_token_dict["connect"], "select": bone_token_dict['select']}
        sub_sel = {"connect": seq_token_dict['connect'], "select": seq_token_dict['select']}

        return {"logits": logits, "feature": tokens, "top_sel": top_sel, "sub_sel": sub_sel}