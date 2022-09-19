import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-12


#Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super().__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            uf = features.unsqueeze(dim=2)
            ua = attentions.unsqueeze(dim=1)
            AiF = uf * ua
            feature_matrix = self.pool(AiF).view(B, -1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class BaseConv2d(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, bn=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(BaseConv2d, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class WSDHead(nn.Module):
    def __init__(self, feature_size, num_classes, num_cparts=6):
        super(WSDHead, self).__init__()
        self.attentions = nn.Sequential(*[
            BaseConv2d(feature_size, feature_size//2, act=True),
            BaseConv2d(feature_size//2, num_cparts, act=True)]
        )
        self.bap = BAP()
        self.att_fc = nn.Linear(feature_size * num_cparts, num_classes, bias=False)

    def forward(self, features):
        batch_size = features.size(0)

        attention_maps = self.attentions(features)
        raw_feature = self.bap(features, attention_maps)

        feature = 100 * raw_feature
        p = self.att_fc(feature.view(batch_size, -1))

        return p, raw_feature, attention_maps