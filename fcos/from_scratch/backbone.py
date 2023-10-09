import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

class Backbone(nn.Module):
    """
    Need to extract 3 layers from resnet

    As shown in model architecture, C3, C4, C5

    - layer 1: 512,  H/8, W/8
    - layer 2: 1028, H/16, W/16
    - layer 3: 2048, H/32, W/32

    """
    def __init__(self):
        super().__init__()
        self.resnet = resnet50()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        layer_1 = self.resnet.layer1(x)
        layer_2 = self.resnet.layer2(layer_1)
        layer_3 = self.resnet.layer3(layer_2)
        layer_4 = self.resnet.layer4(layer_3)
        return layer_2, layer_3, layer_4 # C3, C4, C5

def main():
    resnet50 = Backbone()
    print(resnet50.resnet)

    x = torch.randn((1, 3, 512, 512))

    c3, c4, c5 = resnet50(x)

    print(c3.shape)
    print(c4.shape)
    print(c5.shape)

if __name__ == "__main__":
    main()