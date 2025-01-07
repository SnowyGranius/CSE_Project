import torch
from torch import nn

# DIFFERENT KERNEL SIZES

class NoPoolCNN11(nn.Module): # first attempt at a no pooling CNN - 64 feature maps
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
            nn.Linear(14400 + 3, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
    def forward(self, x, minkowski):
        # Forward through convolutional layers
        x = self.conv_layers(x)
        
        # Concatenate the additional variables
        x = torch.cat((x, minkowski), dim=1)  # Concatenate along the feature dimension
        
        # Forward through fully connected layers
        x = self.fc_layers(x)
        return x
    

class NoPoolCNN12(nn.Module): # first attempt at a no pooling CNN - 64 feature maps
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

            nn.Flatten(),
            nn.Linear(14400, 200),
            nn.ReLU(),
           )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(200 + 3, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
    def forward(self, x, minkowski):
        # Forward through convolutional layers
        x = self.conv_layers(x)
        
        # Concatenate the additional variables
        x = torch.cat((x, minkowski), dim=1)  # Concatenate along the feature dimension
        
        # Forward through fully connected layers
        x = self.fc_layers(x)
        return x
    

class NoPoolCNN13(nn.Module): # first attempt at a no pooling CNN - 64 feature maps
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

            nn.Flatten(),
            nn.Linear(14400, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.ReLU(),
           )
        
        self.fc_layers = nn.Sequential(
            
            nn.Linear(10 + 3, 1)
        )
        
    def forward(self, x, minkowski):
        # Forward through convolutional layers
        x = self.conv_layers(x)
        
        # Concatenate the additional variables
        x = torch.cat((x, minkowski), dim=1)  # Concatenate along the feature dimension
        
        # Forward through fully connected layers
        x = self.fc_layers(x)
        return x
    
class NoPoolCNN14(nn.Module): # first attempt at a no pooling CNN - 64 feature maps
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

        self.mink_layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(14400 + 32, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
    def forward(self, x, minkowski):
        # Forward through convolutional layers and minkowski layer
        x = self.conv_layers(x)
        minkowski = self.mink_layers(minkowski)
        
        # Concatenate the additional variables
        x = torch.cat((x, minkowski), dim=1)  # Concatenate along the feature dimension
        
        # Forward through fully connected layers
        x = self.fc_layers(x)
        return x
    
class NoPoolCNN15(nn.Module): # first attempt at a no pooling CNN - 64 feature maps
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

            nn.Flatten(),
            nn.Linear(14400, 200),
            nn.ReLU()
           )

        self.mink_layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(200 + 32, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
    def forward(self, x, minkowski):
        # Forward through convolutional layers and minkowski layer
        x = self.conv_layers(x)
        minkowski = self.mink_layers(minkowski)
        
        # Concatenate the additional variables
        x = torch.cat((x, minkowski), dim=1)  # Concatenate along the feature dimension
        
        # Forward through fully connected layers
        x = self.fc_layers(x)
        return x
    
class NoPoolCNN16(nn.Module): # first attempt at a no pooling CNN - 64 feature maps
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

            nn.Flatten(),
            nn.Linear(14400, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.ReLU()
           )

        self.mink_layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(10 + 32, 1)
        )
        
    def forward(self, x, minkowski):
        # Forward through convolutional layers and minkowski layer
        x = self.conv_layers(x)
        minkowski = self.mink_layers(minkowski)
        
        # Concatenate the additional variables
        x = torch.cat((x, minkowski), dim=1)  # Concatenate along the feature dimension
        
        # Forward through fully connected layers
        x = self.fc_layers(x)
        return x