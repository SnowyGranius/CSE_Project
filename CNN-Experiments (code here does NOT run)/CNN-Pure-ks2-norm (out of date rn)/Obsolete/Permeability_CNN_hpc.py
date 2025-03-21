import matplotlib.pyplot as plt
import random
import numpy as np
import os
import sys
import pandas as pd
import glob
import re
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import torch
from torch import nn
import torch.nn.functional as F
#from torchsummary import summary
import time


# NLLLoss function expects float64 dtype
torch.set_default_dtype(torch.float64)

my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(my_device)

csv_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'Porespy_homogenous_diamater')
# csv_directory = 'C:\\Users\\ionst\\Documents\\GitHub\\Porespy_homogenous_diamater'
if not os.path.exists(csv_directory):
    raise FileNotFoundError(f"Directory {csv_directory} does not exist.")

#open all csv files in the directory and for each file store the packing fraction found in the name, the shape found in the name and the data in the file as a pandas dataframe
data_csv = []
data_images = []
all_csv = glob.glob(os.path.join(csv_directory, "*.csv"))

for file in all_csv:
    if 'circle' in file:
        df = pd.read_csv(file)
        packing_fraction = re.search(r'\d.\d+', file).group()
        shape = re.search(r'circle|ellipse|trinagle|rectangle', file).group()
        df['Packing_Fraction'] = packing_fraction
        df['Shape'] = shape
        df['Model'] = df.index + 1  # Model number is one higher than the index of the dataframe
        data_csv.append(df)

# Extract permeability values from data_csv
permeability_values = []
for df in data_csv:
    permeability_values.extend(df['Permeability'].values)

print(len(permeability_values))

permeability_values = np.array(permeability_values)
# Scale the permeability values using mean and variance
mean_permeability = np.mean(permeability_values)
std_permeability = np.std(permeability_values)
permeability_values = (permeability_values - mean_permeability) / std_permeability

# value1 = 3.15602165
# value2 = 2.58321055

# Read images from the folder
# Full_Images
# Top_Left_Scales_Images
image_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'Image_dataset_generation/Full_Images_Double')
if not os.path.exists(image_directory):
    raise FileNotFoundError(f"Directory {image_directory} does not exist.")

all_images = glob.glob(os.path.join(image_directory, "*.png"))
# print(all_images)
for image_file in all_images:
    match = re.search(r'pf_(\d\.\d+)_(circle)_Model_(\d+)\.png', image_file)
    if match:
        packing_fraction = float(match.group(1))
        shape = match.group(2)
        model_number = int(match.group(3))
        image = plt.imread(image_file)
        if image.ndim == 3:
            # image = image[:, :, :3]  # Ignore the alpha channel
            image = image[:, :, 0]  # Take only the first channel
        
        data_images.append({
            # 'Model': model_number,
            # 'Packing_Fraction': packing_fraction,
            # 'Shape': shape,
            'Image': image
        })

num_rows = sum(len(df) for df in data_csv)
print(f'Number of rows in data_csv: {num_rows}')


class PermeabilityDataset(torch.utils.data.Dataset):
    '''
    Permeability dataset class for regression
    '''
    def __init__(self, X, y):
        # Precomputed training set statistics for scaling
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            # X is of shape [N, H, W], the CNN expects input [N, Channels, H, W]
            # The image is greyscale (contains only 1 channel), add a dummy channel dimension
            X = np.expand_dims(X, axis=1)
            # Convert to Pytorch tensor objects
            self.X = torch.tensor(X)
            self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
    # This is the method that returns the input, target data during model training
        return self.X[i], self.y[i]


class PermeabilityCNN_COPY(nn.Module):
    def __init__(self):
        super(PermeabilityCNN_COPY, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # pooling has no learnable parameters, so we can just use one
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # MLP classifier
        self.fc = nn.Linear(28800, 1)

    def forward(self, x):
        # print("Input size:", x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv3(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv4(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv5(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv6(x)))
        # print("Layer size:", x.size())
        x = x.view(-1, 28800)  # Flatten the tensor
        # print("Layer size:", x.size())

        # Fully connected layer for classification
        x = self.fc(x)

        return x

class PermeabilityCNN(nn.Module):
    """
    CNN model for classification task on MNIST dataset
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ############## CONV LAYERS #############
            # Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            # Input image has dimensions: [N, C, H, W]
            # 2D convolution layers are defined using the following hyperparameters

            nn.MaxPool2d(kernel_size=5, stride=5),

            nn.Conv2d(
                in_channels = 1,
                out_channels = 32,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ), 
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
            nn.Linear(18432, 1), # 4608, 18432, 51200,
        )
        
    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)

# Helper function to count number of learnable parameters in neural network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Set fixed random number seed for reproducibility of random initializations
torch.manual_seed(42)

torch.set_default_dtype(torch.float32)
#summary(PermeabilityCNN().to(my_device), input_size=(1, 2000, 2000))
torch.set_default_dtype(torch.float64)
data_images = [np.array(image['Image'], dtype=np.float64) for image in data_images]

# Initialize the dataset objects
train_images, dummy_images, train_permeability, dummy_permeability = train_test_split(
    data_images, permeability_values, test_size=0.30, random_state=42)

test_images, val_images, test_permeability, val_permeability = train_test_split(
    dummy_images, dummy_permeability, test_size=0.50, random_state=42)



dataset_train = PermeabilityDataset(X=train_images, y=train_permeability)
dataset_test = PermeabilityDataset(X=test_images, y=test_permeability)
dataset_val = PermeabilityDataset(X=val_images, y=val_permeability)

# Initialize the dataloader using batch size hyperparameter
batch_size = 32
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# Initialize the CNN
cnn = PermeabilityCNN()
cnn_copy = PermeabilityCNN_COPY()
# print(cnn)
# print(f'Number of learnable parameters in {cnn._get_name()} model = {count_parameters(cnn)}')
# Transfer model to your chosen device
cnn.to(my_device)

# Define the loss function and optimizer
## Neg-log-likelihood loss for classification task
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=3e-4)

# Log loss and accuracy per epoch
loss_per_epoch = []
R_squared_per_epoch = []

# Number of epochs to train
n_epochs = 30

# Run the training loop
start_time = time.time()
for epoch in range(0, n_epochs): # n_epochs at maximum

    # Print epoch
    print(f'Starting epoch {epoch+1}')

    # Log loss and accuracy per batch
    loss_per_batch = []
    R_squared_per_batch = []

    # Iterate over batches of training data using the DataLoader
    for i, data in enumerate(trainloader, 0):

        # Get and prepare inputs by converting to floats
        inputs, targets = data
        # Transfer data to your chosen device
        inputs = inputs.to(my_device)
        targets = targets.to(my_device)

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = cnn(inputs)
        outputs = outputs.squeeze()

        # Compute loss
        loss = loss_function(outputs, targets)
        
        # Compute error (absolute difference) for accuracy calculation
        error = torch.abs(outputs - targets)
        # Compute R^2 score
        ss_residual = torch.sum((targets - outputs)**2)
        ss_total = torch.sum((targets - torch.mean(targets))**2)
        # print(ss_residual, ss_total)
        R_squared = 1 - (ss_residual / ss_total)

        # Perform backwards pass
        loss.backward()

        # Perform optimization
        optimizer.step()
        
        # Log loss value per batch
        loss_per_batch.append(loss.item())
        R_squared_per_batch.append(R_squared.item())

    # Log loss value per epoch
    loss_per_epoch.append(np.mean(loss_per_batch))
    
    # Log accuracy value per epoch
    R_squared_per_epoch.append(np.mean(R_squared_per_batch))
    # torch.save(cnn.state_dict(), 'best_model.pth')
    print(f'\tAfter epoch {epoch+1}: Loss = {loss_per_epoch[epoch]}, R-squared = {R_squared_per_epoch[epoch]}')
    
    
# Process is complete.
print('Training process has finished.')
end_time = time.time()

print('Training time: ', end_time-start_time, ' seconds')

#scale back the permeability values
train_permeability = train_permeability * std_permeability + mean_permeability
test_permeability = test_permeability * std_permeability + mean_permeability


fig, axs = plt.subplots(2,1,figsize=(8,8))
# Plot loss per epoch
axs[0].plot(np.arange(1,len(loss_per_epoch)+1), loss_per_epoch, color='blue', label='Training loss', marker='.')
axs[0].grid(True)
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend() 
# Plot accuracy per epoch
axs[1].plot(np.arange(1,len(R_squared_per_epoch)+1), R_squared_per_epoch, color='blue', label='Training R squared', marker='x')
axs[1].grid(True)
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('R squared')
axs[1].legend()
plt.suptitle('Loss and R squared curves during training', y=0.92)
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'Loss_R_squared.png'))
plt.show()

# Evaluate model
cnn.eval()

with torch.no_grad():
    test_inputs = dataset_test.X.to(my_device)
    outputs = cnn(test_inputs)

test_predictions = outputs.cpu().numpy()
test_predictions = test_predictions * std_permeability + mean_permeability
test_targets = np.array(dataset_test.y) * std_permeability + mean_permeability
# Identify outliers in the dataset

# Calculate z-scores of the test targets
z_scores = zscore(test_targets)

# Define a threshold for identifying outliers
threshold = 2

# Identify outliers
outliers = np.where(np.abs(z_scores) > threshold)[0]

print(f'Number of outliers: {len(outliers)}')
print(f'Outliers indices: {outliers}')
print(f'Outliers values: {test_targets[outliers]}')




# Visualize the ground truth on x axis and predicted values on y axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(test_targets, test_predictions, color='blue', label='Predictions')
ax.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], 'k--', lw=2, label='Ideal')
ax.set_xlabel('Ground Truth')
ax.set_ylabel('Predicted')
ax.set_title('Ground Truth vs Predicted Values')
ax.legend()
# r_squared = 1 - np.sum((test_targets - test_predictions)**2) / np.sum((test_targets - np.mean(test_targets))**2)
ax.text(0.05, 0.95, f'R^2: {R_squared_per_epoch[epoch]:.2f}', transform=ax.transAxes, fontsize=14, verticalalignment='top')
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'Ground_Truth_vs_Predicted.png'))
plt.show()
