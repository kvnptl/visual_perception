import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

    def forward(self, x):
        # TODO: replace with single line return
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


# class Block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                       kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
#                       kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.conv(x)


# Check Block
# encoder_block = Block(1, 64)
# x = torch.randn(1, 1, 512, 512)

# LEARNING FROM HERE
'''
torch.rand vs torch.randn

- torch.rand: generates random numbers between [0, 1) with uniform distribution
- torch.randn: generates numbers with normal distribution with mean = 0 and std = 1
(both takes shape as input)
'''

# out = encoder_block(x)
# print(f'Shape of out after encoder block: {out.shape}')


# class Encoder(nn.Module):
#     def __init__(self, channels=(1, 64, 128, 256, 512)):
#         super().__init__()
#         self.encoder_block = nn.ModuleList(
#             [Block(in_channels=channels[i], out_channels=channels[i + 1])
#              for i in range(len(channels) - 1)]
#         )
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         skip_connections = []

#         for idx, block in enumerate(self.encoder_block):
#             x = block(x)
#             skip_connections.append(x)
#             if idx < len(self.encoder_block) - 1:
#                 x = self.pool(x)

#         # print(f'Shape of x after encoder: {x.shape}')

#         return skip_connections


# Check Encoder
# encoder = Encoder(channels=(1, 64, 128, 256))
# x = torch.randn(1, 1, 512, 512)
# skip_connections = encoder(x)
# for i, skip_connection in enumerate(skip_connections):
#     print(f'Shape of skip connection {i + 1}: {skip_connection.shape}')


# class Decoder(nn.Module):
#     def __init__(self, channels=(512, 256, 128, 64)):
#         super().__init__()
#         self.channels = channels
#         self.decoder_block = nn.ModuleList(
#             [Block(in_channels=channels[i], out_channels=channels[i + 1])
#              for i in range(len(channels) - 1)]
#         )
#         self.upconv = nn.ModuleList(
#             [nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=2, stride=2)
#              for i in range(len(channels) - 1)]
#         )

#     def forward(self, x, skip_connections):
#         for i in range(len(self.channels) - 1):
#             x = self.upconv[i](x)
#             skip_connection = skip_connections.pop()
#             # dim = 1 is the channel dimension

#             if x.shape != skip_connection.shape:
#                 x = TF.resize(x, size=skip_connection.shape[2:])

#             x = torch.cat([skip_connection, x], dim=1)
#             x = self.decoder_block[i](x)
#         return x


# Check Decoder
# decoder = Decoder(channels=(512, 256, 128, 64))
# x = torch.randn(1, 512, 64, 64)
# decoder_out = decoder(x, skip_connections)
# print(f'\nShape of decoder output: {decoder_out.shape}')


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder part
        for feature in features:
            self.encoder.append(Block(in_channels, feature))
            in_channels = feature

        # Decoder part
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(
                feature*2, feature, kernel_size=2, stride=2))
            self.decoder.append(Block(feature*2, feature))

        self.bottleneck = Block(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)

            # NOTE: keeping it as raw logits only. BCELogitsLoss would take care of it.
            # Keep in mind that while inference, we need to apply sigmoid on output.
        return self.final_conv(x)


def main():
    # Check UNet
    unet_model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 161, 161)
    print(f'\nShape of input: {x.shape}')
    out = unet_model(x)
    print(f'\nShape of input: {x.shape}')
    print(f'Shape of UNet out: {out.shape}')

    from torchinfo import summary

    summary(unet_model, input_size=(1, 3, 161, 161))


if __name__ == '__main__':
    main()
