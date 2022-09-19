import torch.nn as nn
from .layers import WSDHead
from .utils import ACropDrop
from enum import Enum

class WSDREF(Enum):
    NONE = 0
    CROP = 1
    CROP_DROP = 2


class WSDAN(nn.Module):
    def __init__(self, feature_size, num_classes=1000, num_cparts=32):
        super(WSDAN, self).__init__()

        self.num_classes = num_classes
        self.crop_drop = ACropDrop()
        self.wsd_head = WSDHead(feature_size, num_classes, num_cparts)
    
    def _extract_features(self, images):
        return None
    
    def __one_call(self, images):
        features = self._extract_features(images)
        p, raw_feature, attention_maps = self.wsd_head(features) 
        return {"logits": p, "feature": raw_feature, "attention": attention_maps}
    
    def forward(self, images, rtype=WSDREF.NONE):
        crop_drop = ACropDrop()
        result = self.__one_call(images)
        attention = result.pop('attention')

        if rtype == WSDREF.CROP:
            crop_imgs = crop_drop(images, attention, False) 
            crop_result = self.__one_call(crop_imgs)
            crop_result.pop('attention')

            return {"raw": result, "crop": crop_result} 
        elif rtype == WSDREF.CROP_DROPS:
            crop_imgs, drop_imgs = crop_drop(images, attention, True)
            crop_result = self.__one_call(crop_imgs)
            drop_result = self.__one_call(drop_imgs)

            crop_result.pop('attention')
            drop_result.pop('attention')
            return {"raw": result, "crop": crop_result, "drop": drop_result} 
        else:
            return {"raw": result}