import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import accuracy_score
import os
import sys
import pandas as pd
import glob
import re
from sklearn.model_selection import train_test_split

# NLLLoss function expects float64 dtype
torch.set_default_dtype(torch.float64)

my_device = torch.device('cuda')

csv_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'Porespy_homogenous_diamater')
# csv_directory = 'C:\\Users\\ionst\\Documents\\GitHub\\Porespy_homogenous_diamater'
if not os.path.exists(csv_directory):
    raise FileNotFoundError(f"Directory {csv_directory} does not exist.")
print(csv_directory)

#open all csv files in the directory and for each file store the packing fraction found in the name, the shape found in the name and the data in the file as a pandas dataframe
data_csv = []
data_images = []
all_csv = glob.glob(os.path.join(csv_directory, "*.csv"))
for file in all_csv:
    packing_fraction = re.search(r'\d\.\d+', file).group()
    shape = re.search(r'circle|ellipse|rectangle|triangle', file).group()
    df = pd.read_csv(file)
    df['Packing_Fraction'] = packing_fraction
    df['Shape'] = shape
    df['Model'] = df.index + 1  # Model number is one higher than the index of the dataframe
    data_csv.append(df)

    # Extract permeability values from data_csv
    permeability_values = []
    for df in data_csv:
        permeability_values.extend(df['Permeability'].values)

    permeability_values = np.array(permeability_values)
#print(data_csv)
print(permeability_values)

# Read images from the folder
image_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'Image_dataset_generation/Circle_Images')
if not os.path.exists(image_directory):
    raise FileNotFoundError(f"Directory {image_directory} does not exist.")

all_images = glob.glob(os.path.join(image_directory, "*.png"))
# print(all_images)
for image_file in all_images:
    match = re.search(r'pf_(\d\.\d+)_(circle|ellipse|rectangle|triangle)_Model_(\d+)\.png', image_file)
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

print(data_images)
# Transform data_images to float64 format
data_images = [np.array(image['Image'], dtype=np.float64) for image in data_images]
print(len(data_images))
# Print the number of rows in data_csv
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
            nn.Conv2d(
                in_channels = 1,
                out_channels = 10,
                kernel_size = 5,
                stride = 1,
            ), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            ############## MLP LAYERS #############
            nn.Linear(1220180, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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

# Initialize the dataset objects
train_images, test_images, train_permeability, test_permeability = train_test_split(
    data_images, permeability_values, test_size=0.30, random_state=42)

dataset_train = PermeabilityDataset(X=train_images, y=train_permeability)
dataset_test = PermeabilityDataset(X=test_images, y=test_permeability)

# Initialize the dataloader using batch size hyperparameter
batch_size = 32
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# Initialize the CNN
cnn = PermeabilityCNN()
print(cnn)
print(f'Number of learnable parameters in {cnn._get_name()} model = {count_parameters(cnn)}')
# Transfer model to your chosen device
cnn.to(my_device)

# Define the loss function and optimizer
## Neg-log-likelihood loss for classification task
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)

# Log loss and accuracy per epoch
loss_per_epoch = []
acc_per_epoch = []

# Number of epochs to train
n_epochs = 10

# Run the training loop
for epoch in range(0, n_epochs): # n_epochs at maximum

    # Print epoch
    print(f'Starting epoch {epoch+1}')

    # Log loss and accuracy per batch
    loss_per_batch = []
    acc_per_batch = []

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

        # Compute loss
        loss = loss_function(outputs, targets)
        
        # Compute error (absolute difference) for accuracy calculation
        error = torch.abs(outputs - targets)
        accuracy = 1 - torch.mean(error).item()

        # Perform backwards pass
        loss.backward()

        # Perform optimization
        optimizer.step()
        
        # Log loss value per batch
        # loss_per_batch.append(loss.item())
        acc_per_batch.append(accuracy)

    # Log loss value per epoch
    loss_per_epoch.append(np.mean(loss_per_batch))
    
    # Log accuracy value per epoch
    acc_per_epoch.append(np.mean(acc_per_batch))
    
    print(f'\tAfter epoch {epoch+1}: Loss = {loss_per_epoch[epoch]}, Accuracy = {acc_per_epoch[epoch]}')
    
    
# Process is complete.
print('Training process has finished.')

fig, axs = plt.subplots(2,1,figsize=(8,8))
# Plot loss per epoch
axs[0].plot(np.arange(1,len(loss_per_epoch)+1), loss_per_epoch, color='blue', label='Training loss', marker='.')
axs[0].grid(True)
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Neg-log-likelihood Loss')
axs[0].legend()
# Plot accuracy per epoch
axs[1].plot(np.arange(1,len(acc_per_epoch)+1), acc_per_epoch, color='blue', label='Training accuracy', marker='x')
axs[1].grid(True)
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend()
plt.suptitle('Loss and accuracy curves during training', y=0.92)
plt.show()

# Evaluate model
cnn.eval()

with torch.no_grad():
    test_inputs = dataset_test.X.to(my_device)
    outputs = cnn(test_inputs)
    predicted_labels = torch.argmax(outputs, dim=1).cpu()

test_predictions = np.array(predicted_labels)
test_targets = np.array(dataset_test.y)

test_accuracy = accuracy_score(test_predictions, test_targets)

print(f'Accuracy Score on test set: {test_accuracy}')


# Helper function to visualize the MNIST image, ground truth and predicted labels on a subplot axis
def visualize_predictions(x,y,y_pred,ax):
    ax.imshow(x, cmap='gray')
    ax.set(
        title = f'Ground truth: {y}\nPrediction: {y_pred}',
        xticks = [],
        yticks = [],
    )

fig, axes = plt.subplots(2,4, layout='tight')
axs = axes.flatten()
random_ids = random.sample(list(range(x_test.shape[0])), 8)
for i, id in enumerate(random_ids):
    axs[i] = visualize_predictions(test_inputs[id,0,...].cpu(), test_targets[id], test_predictions[id], axs[i])