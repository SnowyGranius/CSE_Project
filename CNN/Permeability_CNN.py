import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import os
import sys
import pandas as pd
import glob
import re
from sklearn.model_selection import train_test_split
from torchsummary import summary
from scipy.stats import zscore
from classes_cnn import BasicCNN, MLPCNN, NoPoolCNN1, NoPoolCNN2, EvenCNN, EvenCNN2000
import time


# NLLLoss function expects float64 dtype
torch.set_default_dtype(torch.float64)

validation = False

my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {my_device}')

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

# Read images from the folder
# Full_Images
# Top_Left_Scaled_Images
# Full_Images_Double
image_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'Image_dataset_generation/Top_Left_Scaled_Images')
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



# Helper function to count number of learnable parameters in neural network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Set fixed random number seed for reproducibility of random initializations
torch.manual_seed(42)


data_images = [np.array(image['Image'], dtype=np.float64) for image in data_images]

# Initialize the dataset objects



if validation:
    train_images, dummy_images, train_permeability, dummy_permeability = train_test_split(
        data_images, permeability_values, test_size=0.30, random_state=42)

    test_images, val_images, test_permeability, val_permeability = train_test_split(
        dummy_images, dummy_permeability, test_size=0.50, random_state=42)

else:
    train_images, test_images, train_permeability, test_permeability = train_test_split(
        data_images, permeability_values, test_size=0.30, random_state=42)

# normalize the training permeability
mean_permeability = np.mean(train_permeability)
std_permeability = np.std(train_permeability)
train_permeability = (train_permeability - mean_permeability) / std_permeability
test_permeability = (test_permeability - mean_permeability) / std_permeability

dataset_train = PermeabilityDataset(X=train_images, y=train_permeability)
dataset_test = PermeabilityDataset(X=test_images, y=test_permeability)

if validation:
    val_permeability = (val_permeability - mean_permeability) / std_permeability
    dataset_val = PermeabilityDataset(X=val_images, y=val_permeability)


# Initialize the dataloader using batch size hyperparameter
batch_size = 32
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

# Initialize the CNN
loss_function = nn.MSELoss()

for cnn in [NoPoolCNN2().to(my_device), NoPoolCNN1().to(my_device)]:
    for lr in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:

        torch.set_default_dtype(torch.float32)
        summary(NoPoolCNN2().to('cuda'), input_size=(1, 1000, 1000))
        torch.set_default_dtype(torch.float64)

        optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

        # Log loss and accuracy per epoch
        loss_per_epoch = []
        R_squared_per_epoch = []

        # Number of epochs to train
        n_epochs = 50

        start_time = time.time()
        # Run the training loop
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
            print(f'\tAfter epoch {epoch+1}: Loss = {loss_per_epoch[epoch]}, R-squared = {R_squared_per_epoch[epoch]}')
            
            
        # Process is complete.
        print('Training process has finished.')

        end_time = time.time()
        print(f'Training time: {end_time - start_time} seconds')

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
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Loss_R_squared-{cnn.__class__.__name__}-{lr}.png'))

        # Evaluate model
        cnn.eval()

        torch.cuda.empty_cache()

        with torch.no_grad():
            test_inputs = dataset_test.X.to('cpu')
            cnn = cnn.to('cpu')
            test_predictions = cnn(test_inputs).cpu().numpy()
            data_images_np = np.array(data_images)
            data_images_tensor = torch.tensor(data_images_np).unsqueeze(1).to('cpu')
            all_predictions = cnn(data_images_tensor).cpu().numpy()


        # test_predictions = outputs_test.cpu().numpy()
        test_predictions = test_predictions * std_permeability + mean_permeability
        test_targets = np.array(dataset_test.y) * std_permeability + mean_permeability
        # all_predictions = outputs_all.cpu().numpy()
        all_predictions = all_predictions * std_permeability + mean_permeability
        all_targets = permeability_values * std_permeability + mean_permeability

        # compute the test R squared
        ss_residual = np.sum((test_targets - test_predictions)**2)
        ss_total = np.sum((test_targets - np.mean(test_targets))**2)
        R_squared = 1 - (ss_residual / ss_total)


        # Visualize the ground truth on x axis and predicted values on y axis
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(test_targets, test_predictions, color='blue', label='Predictions')
        ax.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], 'k--', lw=2, label='Ideal')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title('Ground Truth vs Predicted Values')
        ax.legend()
        # r_squared = 1 - np.sum((test_targets - test_predictions)**2) / np.sum((test_targets - np.mean(test_targets))**2)
        ax.text(0.05, 0.95, f'Test R^2: {R_squared:.2f}', transform=ax.transAxes, fontsize=14, verticalalignment='top')
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Ground_Truth_vs_Predicted-{cnn.__class__.__name__}-{lr}.png'))
        #free up memory
        del test_inputs
        del data_images_np
        del data_images_tensor
        del test_predictions
        del test_targets
        del all_predictions
        del all_targets
        del R_squared
        torch.cuda.empty_cache()
        cnn = cnn.to(my_device) 