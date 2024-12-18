import torch
from torch import nn

# DIFFERENT KERNEL SIZES

class NoPoolCNN11(nn.Module): # first attempt at a no pooling CNN - 64 feature maps
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(

            nn.Conv2d(1, 8, kernel_size=2, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten()
           )
        
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(7744 + 3, 200),  # Adjust input size to include 3 additional variables
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

            nn.Conv2d(1, 8, kernel_size=2, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(7744, 200), # 4 = 3136, 5 = 4096, 6 = 6400, 7 = 7744, 8 = 10816, 9 = 14400, 10 = 16384
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

            nn.Conv2d(1, 8, kernel_size=2, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(7744, 200), # 4 = 3136, 5 = 4096, 6 = 6400, 7 = 7744, 8 = 10816, 9 = 14400, 10 = 16384
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
        


class NoPoolCNN3(nn.Module): # heavier model compared to 1 - 128 feature maps
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(

            nn.Conv2d(1, 8, kernel_size=2, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            

            nn.Flatten(),
            nn.Linear(3136*2, 200), # 4 = 3136*2, 5 = 4096*2, 6 = 6400*2, 7 = 7744*2, 8 = 10816*2, 9 = 14400*2, 10 = 16384*2
            nn.ReLU(),
            nn.Linear(200, 10),
            # activation
            nn.Linear(10, 1)
        )
        
    def forward(self, x):
        return self.layers(x)
    
class NoPoolCNN4(nn.Module): # heaviest model - 256 feature maps
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(

            nn.Conv2d(1, 8, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Flatten(),
            nn.Linear(3136*4, 200), # 4 = 3136*4, 5 = 4096*4, 6 = 6400*4, 7 = 7744*4, 8 = 10816*4, 9 = 14400*4, 10 = 16384*4
            nn.ReLU(),
            nn.Linear(200, 10),
            # activation
            nn.Linear(10, 1)
        )
        
    def forward(self, x):
        return self.layers(x)