import sys

sys.path.insert(0, '../src')

from dnets.cclip import ClipEmbedding


net = ClipEmbedding('./cclip/conf.yaml')

input = {'img': './classify/hashiqi.jpg', 'txt': "hashiqi"}

res = net(input)

print(res['img_embedding'].shape)
print(res['txt_embedding'].shape)