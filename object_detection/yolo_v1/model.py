import torch
import torch.nn as nn

architecture_config = [
    (7, 64, 2, 3), # kernel, channels, stride, padding
    (2, 2), # kernel, stride

    (3, 192, 1, 1),
    (2, 2),

    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    (2, 2),
    # List: tuple(kernel, channels, stride, padding) and last one is the number of repeats
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # 0
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    (2, 2),

    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             bias=False,
                             **kwargs)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YOLOv1(nn.Module):
    def __init__(self, in_channels: int=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fcs = self.create_fcs(**kwargs)

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            # Conv block
            if type(x) == tuple and len(x) == 4:
                layers += [CNNBlock(in_channels=in_channels, 
                                  out_channels=x[1], 
                                  kernel_size=x[0], 
                                  stride=x[2], 
                                  padding=x[3])]
                in_channels = x[1]
                
            # Maxpool
            elif type(x) == tuple and len(x) == 2:
                layers += [nn.MaxPool2d(kernel_size=x[0], stride=x[1])]
            # Conv repeated block
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [CNNBlock(in_channels=in_channels, 
                                      out_channels=conv1[1], 
                                      kernel_size=conv1[0], 
                                      stride=conv1[2], 
                                      padding=conv1[3])]
                    layers += [CNNBlock(in_channels=conv1[1], 
                                      out_channels=conv2[1], 
                                      kernel_size=conv2[0], 
                                      stride=conv2[2], 
                                      padding=conv2[3])]
                    in_channels = conv2[1]

        return nn.Sequential(*layers) # *layers unpacks the list of layers
    
    def create_fcs(self, split_size=7, num_boxes=2, num_classes=20): # split_size is the grid size
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024 * S * S, out_features=4096),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=4096, out_features=S*S*(C + B*5)), # each cell is 30x1 (20 classes + (1+4) 1st box + (1+4) 2nd box), where (1+4): probability + x1, y1, w, h
        )

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(x)

def main():
    model = YOLOv1()
    x = torch.randn((1, 3, 448, 448))
    
    from torchinfo import summary

    summary(model=model,
            input_size=x.shape, # (batch_size, channels, height, width)
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]) 

if __name__ == "__main__":
    main()