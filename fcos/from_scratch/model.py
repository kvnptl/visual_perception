
"""

Slightly modified FCOS from Ross Hemsley's implementation.

Source: https://github.com/rosshemsley/fcos

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import numpy as np

from backbone import Backbone

class FCOS(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = Backbone()

        self.scales = nn.Parameter(torch.Tensor([8, 16, 32, 64, 128])) # what is this?
        self.strides = [8, 16, 32, 64, 128]

        # Feature Pyramid Network
        self.c3_to_p3 = conv_group_norm(in_channels=512, out_channels=256)
        self.c4_to_p4 = conv_group_norm(in_channels=1024, out_channels=256)
        self.c5_to_p5 = conv_group_norm(in_channels=2048, out_channels=256)
        self.p5_to_p6 = conv_group_norm(in_channels=256, out_channels=256, stride=2)
        self.p6_to_p7 = conv_group_norm(in_channels=256, out_channels=256, stride=2)

        self.classification_head = nn.Sequential(
            conv_group_norm(256, 256),
            nn.ReLU(inplace=True),
            conv_group_norm(256, 256),
            nn.ReLU(inplace=True),
            conv_group_norm(256, 256),
            nn.ReLU(inplace=True),
            conv_group_norm(256, 256),
            nn.ReLU(inplace=True),
        )

        self.classification_to_class = nn.Sequential(conv_group_norm(256, self.num_classes))
        self.classification_to_centerness = nn.Sequential(conv_group_norm(256, 1))

        self.regression_head = nn.Sequential(
            conv_group_norm(256, 256),
            nn.ReLU(inplace=True),
            conv_group_norm(256, 256),
            nn.ReLU(inplace=True),
            conv_group_norm(256, 256),
            nn.ReLU(inplace=True),
            conv_group_norm(256, 256),
            nn.ReLU(inplace=True),
            conv_group_norm(256, 4)
        )

        for modules in [
            self.c3_to_p3,
            self.c4_to_p4,
            self.c5_to_p5,
            self.p5_to_p6,
            self.p6_to_p7,
            self.classification_head,
            self.classification_to_class,
            self.classification_to_centerness,
            self.regression_head
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.constant_(layer.bias, 0)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Takes as input a batch of images encoded as BCHW, which have been normalized using
        the function `normalize_batch`.

        Returns a list of tuples, where each entry in the list represents one of the levels of the five
        feature maps. The tuple contains respectively, 
        - A float tensor indexed as BHWC, where C represents the 1-hot encoded class labels.
        - A float tensor indexed as BHW[x_min, y_min, x_max, y_max], where the values correspond
          directly with the input tensor, x
        """

        batch, channels, height, width = x.shape

        c3, c4, c5 = self.backbone(x)
        # Following model architecture
        p5 = self.c5_to_p5(c5)
        p6 = self.p5_to_p6(p5)
        p7 = self.p6_to_p7(p6)
        p4 = self.c4_to_p4(c4) + upsample(p5, size=c4.shape[2:4])
        p3 = self.c3_to_p3(c3) + upsample(p4, size=c3.shape[2:4])

        feature_pyramid = [p3, p4, p5, p6, p7]

        classes_by_feature = []
        centerness_by_feature = []
        regression_by_feature = []

        for scale, stride, feature in zip(self.scales, self.strides, feature_pyramid):
            classification = self.classification_head(feature)
            
            classes = self.classification_to_class(classification).sigmoid()
            centerness = self.classification_to_centerness(classification).sigmoid()
            regression = scale * self.regression_head(feature)

            # B[C]HW -> BHW[C]
            classes = classes.permute(0, 2, 3, 1).contiguous()
            # NOTE: In PyTorch, tensors can be stored in non-contiguous memory, which means the elements are not stored sequentially in memory. This can happen when you do certain operations like permutations, slicing, etc.
            # Non-contiguous memory access is slower because it causes extra memory lookup overhead. So .contiguous() is used to ensure the tensor data is stored in a contiguous chunk of memory to speed up operations.

            centerness = centerness.permute(0, 2, 3, 1).contiguous().squeeze(3)

            regression = regression.permute(0, 2, 3, 1).contiguous()

            classes_by_feature.append(classes)
            centerness_by_feature.append(centerness)
            regression_by_feature.append(regression)

        return classes_by_feature, centerness_by_feature, regression_by_feature

            

def upsample(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)

def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor representing a batch of unnormalized B[RGB]HW images,
    where RGB are floating point values in the range 0 to 255, prepare the tensor for the
    FCOS network. This has been defined to match the backbone resnet50 network.
    See https://pytorch.org/docs/master/torchvision/models.html for more details.
    """
    f = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for b in range(x.shape[0]):
        f(x[b])

    return x


def conv_group_norm(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.GroupNorm(num_groups=32, num_channels=in_channels), # The in_channels dimension is split into 32 even groups. So if in_channels are 128, each group would have 128 / 32 = 4 channels.
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
    )


def main():
    model = FCOS(num_classes=80)

    x = torch.randn((1, 3, 512, 512))

    # Visualize the network
    visualize_network(model, x)

    # classes, centerness, regression = model(x)

    # print(f"Layers: {len(classes)}, Shape: {classes[0].shape}")
    # print(f"Layers: {len(centerness)}, Shape: {centerness[0].shape}")
    
    # for i, val in enumerate(regression):
    #     print(f"Shape of regression layer {i}: {val.shape}")

def visualize_network(model, x):
    # visualize the network
    from torchview import draw_graph
    from torchinfo import summary
    import torchlens

    summary(model=model, 
            input_size=x.shape,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    # # Visualize the network
    # model_graph = draw_graph(model, x, 
    #                          expand_nested=True,
    #                          hide_module_functions=False,
    #                          graph_name='fcos_from_scratch_graph_torchview')
    # model_graph.resize_graph(scale=1.0)
    # model_graph.visual_graph.render(format='png')
    
    # Torchlens
    # torchlens.log_forward_pass(model=model.eval(),
    #                         input_args=x,
    #                         vis_opt='rolled',
    #                         vis_save_only=True,
    #                         vis_outpath='fcos_from_scratch_graph_torchlens',
    #                         vis_fileformat='png',
    #                         vis_direction='topdown')

if __name__ == "__main__":
    main()