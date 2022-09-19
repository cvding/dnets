import torch
import torch.nn.functional as F
import numpy as np
import random


class ACropDrop(object):
    def __init__(self, wh_thr=(0.1, 0.1)):
        self.wh_thr = wh_thr
        #self.wh_pad = (int(wh_thr[0]*w), int(wh_thr[1]*h))

    @torch.no_grad()
    def __call__(self, images, attention_maps, crop_drop_type:bool=True):
        B, _, H, W = images.shape
        _, M, _, _ = attention_maps.shape
        ret_imgs = []
        masks = []
        wh_pad = (int(self.wh_thr[0] * W), int(self.wh_thr[1] * H))

        attention_maps = F.interpolate(attention_maps, size=(W, H), mode='bilinear', align_corners=True)
        if crop_drop_type:
            part_weights = F.avg_pool2d(attention_maps, (W, H)).reshape(B, -1)
            part_weights = torch.add(torch.sqrt(part_weights), 1e-12)
            part_weights = torch.div(part_weights, torch.sum(part_weights, dim=1).unsqueeze(1)).cpu()
            part_weights = part_weights.numpy()

            for i in range(B):
                attention_map = attention_maps[i]
                part_weight = part_weights[i]
                selected_index = np.random.choice(np.arange(0, M), 1, p=part_weight)[0]
                ## create crop imgs
                mask = attention_map[selected_index, :, :]
                threshold = random.uniform(0.4, 0.6)
                itemindex = torch.nonzero(mask >= threshold*mask.max())

                height_min = itemindex[:,0].min()
                height_min = max(0,height_min-wh_pad[1])
                height_max = itemindex[:,0].max() + wh_pad[1]

                width_min = itemindex[:,1].min()
                width_min = max(0,width_min-wh_pad[0])
                width_max = itemindex[:,1].max() + wh_pad[0]

                if height_max - height_min < wh_pad[1] or width_max - width_min < wh_pad[0]:
                    out_img = images[i].unsqueeze(0)
                else:
                    out_img = images[i][:,height_min:height_max,width_min:width_max].unsqueeze(0)
                    out_img = torch.nn.functional.interpolate(out_img,size=(W,H),mode='bilinear',align_corners=True)
                    out_img = out_img.squeeze(0)
                ret_imgs.append(out_img)

                # drop
                selected_index = np.random.choice(np.arange(0, M), 1, p=part_weight)[0]

                mask = attention_map[selected_index:selected_index + 1, :, :]
                threshold = random.uniform(0.2, 0.5)
                mask = (mask < threshold * mask.max()).float()
                masks.append(mask)
            crop_imgs = torch.stack(ret_imgs)
            drop_imgs = images * torch.stack(masks)

            return crop_imgs, drop_imgs

        else:
            threshold = 0.1
            padding_h = int(0.05*H)
            padding_w = int(0.05*W)
            for i in range(B):
                att_map = attention_maps[i]
                mask = att_map.mean(dim=0)
                max_activate = mask.max()
                min_activate = threshold * max_activate
                itemindex = torch.nonzero(mask >= min_activate)
                height_min = itemindex[:, 0].min()
                height_min = max(0, height_min-padding_h)
                height_max = itemindex[:, 0].max() + padding_h
                width_min = itemindex[:, 1].min()
                width_min = max(0, width_min-padding_w)
                width_max = itemindex[:, 1].max() + padding_w
                # print(height_min,height_max,width_min,width_max)
                out_img = images[i][:,height_min:height_max,width_min:width_max].unsqueeze(0)
                out_img = F.interpolate(out_img,size=(W, H), mode='bilinear', align_corners=True)
                out_img = out_img.squeeze(0)
                # print(out_img.shape)
                ret_imgs.append(out_img)
            crop_imgs = torch.stack(ret_imgs)

            return crop_imgs





if __name__ == '__main__':
    # net = ACropDrop()
    # x = torch.randn(5, 3, 5, 5)
    # a = torch.randn(5, 2, 5, 5)
    # a[a<0] = 0
    # a = F.normalize(a, p=2, dim=-1)
    # print(a)
    # #net.eval()
    # y = net(x, a)
    # print(y[0].shape, y[1].shape)