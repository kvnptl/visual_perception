import torch
import torch.nn as nn

from line_profiler import LineProfiler

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def profile(func):
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lp.add_function(func)  # Add the function you want to profile
        result = lp(func)(*args, **kwargs)
        lp.print_stats(output_unit=1e-3)
        return result
    return wrapper

@profile
def main():
    # Define a simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 32 * 8 * 8)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Define your model
    model = SimpleCNN().to(device)

    # Parameters
    batch_size = 16
    input_shape = (3, 32, 32)  # For example, CIFAR-10 dataset input shape
    output_shape = (10,)  # Number of classes in the dataset

    # Generate random input data
    inputs = torch.randn(batch_size, *input_shape).to(device)

    # Generate random ground truth values for the sake of this example
    targets = torch.randint(0, 10, (batch_size,)).to(device)

    # Define your loss function
    loss_function = nn.CrossEntropyLoss()

    # Forward pass: compute predicted outputs by passing inputs to the model
    outputs = model(inputs)

    # Compute the loss
    loss = loss_function(outputs, targets)

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

if __name__ == "__main__":
    main()