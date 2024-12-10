import torch
from torch import nn

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(

            nn.MaxPool2d(kernel_size=5, stride=5),

            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            ############## MLP LAYERS #############
            nn.Linear(4608, 1), # 4608, 18432, 51200,
        )
        
    def forward(self, x):
        return self.layers(x)
    

class MLPCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(

            nn.MaxPool2d(kernel_size=5, stride=5),

            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            ############## MLP LAYERS #############
            nn.Linear(4608, 800), # 4608, 18432, 51200,
            nn.ReLU(),
            nn.Linear(800, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Linear(10, 1)
        )
        
    def forward(self, x):
        return self.layers(x)
    
class NoPoolCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=2, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),

            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            
            nn.Conv2d(64, 128, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            
            nn.Conv2d(128, 256, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Flatten(),
            ############## MLP LAYERS #############
            nn.Linear(25600, 800), # 4608, 18432, 51200,
            nn.ReLU(),
            nn.Linear(800, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Linear(10, 1)
        )
        
    def forward(self, x):
        return self.layers(x)
    
class EvenCNN(nn.Module):
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

            nn.Conv2d(256, 256, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Flatten(),
            ############## MLP LAYERS #############
            nn.Linear(16384, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.layers(x)


    
class EvenCNN2000(nn.Module):
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

            nn.Conv2d(256, 256, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(256, 256, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Flatten(),
            ############## MLP LAYERS #############
            nn.Linear(16384, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.layers(x)