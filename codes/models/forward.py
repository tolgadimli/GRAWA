from pyramidnet import PyramidNet
from resnet import ResNet20
from vgg import VGG16
from wideresnet import Wide_ResNet

import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

x = torch.rand(128,3,32,32).to(device)

model = PyramidNet(110, 270, 10).to(device)
y1 = model(x)
print(y1)

# del model

# model = Wide_ResNet(28, 10, num_classes=10).to(device)
# y2 = model(x)
# print(y2)