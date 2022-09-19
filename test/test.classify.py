import sys

sys.path.insert(0, '../src')

from dnets.classify import GeneralRecognition

net = GeneralRecognition('./classify/conf.yaml')

input = {'img': './classify/hashiqi.jpg'}

res = net(input)

print(res)

