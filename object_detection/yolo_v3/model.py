import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (number of filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_activation=True, **kwargs): # bn_activation = batch normalization and activation
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              bias=not bn_activation, # if we use batch normalization and activation then we don't need bias (???)
                              **kwargs)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        self.use_bn_activation = bn_activation

    def forward(self, x):
        if self.use_bn_activation:
            return self.leakyrelu(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(in_channels=channels, out_channels=channels//2, kernel_size=1),
                    CNNBlock(in_channels=channels//2, out_channels=channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = layer(x) + x
            else:
                x = layer(x)

        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels=in_channels, out_channels=2*in_channels, kernel_size=3, padding=1),
            CNNBlock(in_channels=2*in_channels, out_channels=3*(num_classes+5), bn_activation=False, kernel_size=1), # 3*(num_classes+5) = per cell we have 3 anchors, 5 is for [pc, x, y, w, h]
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2) 
        )
    # N x 3 x 13 x 13 x (num_classes + 5) - this is for scale 1 (13x13) with 3 anchors per cell, N is batch size

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self.create_conv_layers()

    def create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in config:
            if isinstance(module, tuple):
                layers.append(
                    CNNBlock(in_channels=in_channels, 
                            out_channels=module[0], 
                            kernel_size=module[1], 
                            stride=module[2], 
                            padding=1 if module[1] == 3 else 0)
                )
                in_channels = module[0]

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(channels=in_channels,
                                           num_repeats=num_repeats))
                
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(channels=in_channels,
                                      use_residual=False,
                                      num_repeats=1),
                        CNNBlock(in_channels=in_channels,
                                  out_channels=in_channels//2,
                                  kernel_size=1),
                        ScalePrediction(in_channels=in_channels//2,
                                       num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3

        return layers
    
    def forward(self, x):
        outputs = [] # to the scale outputs
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs
    
if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416 # In YOLOv1: (448, 448)
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))

    # Model summary
    from torchinfo import summary

    summary(model=model,
            input_size=x.shape,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")



