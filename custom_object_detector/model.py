from torch import nn
import torchvision
import torch

class ObjectDetector(nn.Module):
    def __init__(self, basemodel, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = basemodel

        self.classifier = nn.Sequential(
            nn.Linear(basemodel.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.num_classes),
        )

        self.regressor = nn.Sequential(
            nn.Linear(basemodel.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )

        self.backbone.fc = nn.Identity()

    def forward(self, x):
        features = self.backbone(x)
        classlogits = self.classifier(features)
        bboxes = self.regressor(features)
        return (classlogits, bboxes)
    

def main():
    # visualize the network
    from torchview import draw_graph
    from torchinfo import summary

    num_classes = 10
    IMAGE_SIZE = 224
    # Create the network
    basemodel = torchvision.models.resnet50(pretrained=True)
    model = ObjectDetector(basemodel, num_classes)

    x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE))    

    summary(model=model, 
            input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    # # Visualize the network
    # model_graph = draw_graph(model, x, graph_name='YOLOv3')
    # model_graph.resize_graph(scale=1.0)
    # model_graph.visual_graph.render(format='svg')

if __name__ == "__main__":
    main()