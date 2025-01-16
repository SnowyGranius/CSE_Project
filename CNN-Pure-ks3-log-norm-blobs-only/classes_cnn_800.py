import torch
from torch import nn

# DIFFERENT KERNEL SIZES

class NoPoolCNN1(nn.Module): # first attempt at a no pooling CNN - 64 feature maps
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(

            nn.Conv2d(1, 8, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten()
           )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(9216, 200),  # Adjust input size to include 3 additional variables
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
    def forward(self, x):
        # Forward through convolutional layers
        x = self.conv_layers(x)
        
        # Forward through fully connected layers
        x = self.fc_layers(x)
        return x
    