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
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels =out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.conv2d = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                bias=False,
                                kernel_size=self.kernel,
                                stride=self.stride,
                                padding=self.padding)
        self.batchnorm = nn.BatchNorm2d(num_features=self.out_channels)
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return self.leakyRelu(self.batchnorm(self.conv2d(x)))

class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, grid_size=7, num_classes=20, num_bbox=2):
        super().__init__()
        self.architectire = architecture_config
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_bbox = num_bbox
        self.backbone_block = self.backbone()
        self.fc_head = self.fch()

    def backbone(self):
        
        in_channels = self.in_channels
        modules = []

        for item in self.architectire:
            if len(item) == 4 and type(item) == tuple:
                cnn_block = CNNBlock(in_channels=in_channels,
                                     out_channels=item[1],
                                     kernel=item[0],
                                     stride=item[2],
                                     padding=item[3])
                in_channels = item[1]
                modules.append(cnn_block)

            elif len(item) == 2 and type(item) == tuple:
                modules.append(nn.MaxPool2d(kernel_size=item[0], stride=item[1]))
            
            elif type(item) == list:
                for i in range(item[-1]):
                    cnn_block1 = CNNBlock(in_channels=in_channels,
                                     out_channels=item[0][1],
                                     kernel=item[0][0],
                                     stride=item[0][2],
                                     padding=item[0][3])
                    in_channels = item[0][1]
                    modules.append(cnn_block1)
                    cnn_block2 = CNNBlock(in_channels=in_channels,
                                     out_channels=item[1][1],
                                     kernel=item[1][0],
                                     stride=item[1][2],
                                     padding=item[1][3])
                    in_channels = item[1][1]
                    modules.append(cnn_block2)

            else:
                raise("Unexpected input")
        
        return nn.Sequential(*modules)


    def fch(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024 * self.grid_size*self.grid_size, out_features=4096),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=4096, out_features=self.grid_size*self.grid_size*(self.num_classes + self.num_bbox*5))
        )

    def forward(self, x):
        x = self.backbone_block(x)
        x = self.fc_head(x)
        return x


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