import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import os
import sys
import pandas as pd
import glob
import re
from sklearn.model_selection import train_test_split
from torchsummary import summary
from scipy.stats import zscore
from classes_cnn import BasicCNN, MLPCNN, NoPoolCNN1, NoPoolCNN2, NoPoolCNN3, NoPoolCNN4, EvenCNN, EvenCNN2000
import time

# Default dype is float64. Not working currently on DelftBlue
torch.set_default_dtype(torch.float64)

# Set fixed random number seed for reproducibility of random initializations
torch.manual_seed(42)

# Choose whether to use take account 15% validation set or not. Not used at the moment
validation = False

# Use CUDA
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {my_device}')

# Load the data from csv files and read the images
csv_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'Porespy_homogenous_diamater')
if not os.path.exists(csv_directory):
    raise FileNotFoundError(f"Directory {csv_directory} does not exist.")

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

permeability_values = []
for df in data_csv:
    permeability_values.extend(df['Permeability'].values)

print(f'Number of permeability values: {len(permeability_values)}')

permeability_values = np.array(permeability_values)

# Read images from the folder
# Full_1000
# Quarter_1000
# Full_2000
image_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'Image_dataset_generation/Quarter_1000')
if not os.path.exists(image_directory):
    raise FileNotFoundError(f"Directory {image_directory} does not exist.")

all_images = glob.glob(os.path.join(image_directory, "*.png"))
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

data_images = [np.array(image['Image'], dtype=np.float64) for image in data_images]
print(f'Number of images: {len(data_images)}')

# Define the dataset class
class PermeabilityDataset(torch.utils.data.Dataset):
    '''
    Permeability dataset class for regression
    '''
    def __init__(self, X, y):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            X = np.expand_dims(X, axis=1)
            self.X = torch.tensor(X)
            self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# Splitting the data into training, testing (and validation) sets

if validation:
    train_images, dummy_images, train_permeability, dummy_permeability = train_test_split(
        data_images, permeability_values, test_size=0.30, random_state=42)

    test_images, val_images, test_permeability, val_permeability = train_test_split(
        dummy_images, dummy_permeability, test_size=0.50, random_state=42)

else:
    train_images, test_images, train_permeability, test_permeability = train_test_split(
        data_images, permeability_values, test_size=0.30, random_state=42)

# Normalizing the training permeability
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
batch_size = int(len(train_permeability)/2)
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

loss_function = nn.MSELoss()

# Visualizing the CNN architectures
torch.set_default_dtype(torch.float32)
summary(NoPoolCNN3().to('cuda'), input_size=(1, 1000, 1000))
summary(NoPoolCNN1().to('cuda'), input_size=(1, 1000, 1000))
summary(NoPoolCNN4().to('cuda'), input_size=(1, 1000, 1000))
torch.set_default_dtype(torch.float64)

for cnn in [NoPoolCNN4().to(my_device), NoPoolCNN3().to(my_device), NoPoolCNN1().to(my_device)]:
    for lr in [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
        print(f'Using CNN: {cnn.__class__.__name__} with learning rate: {lr}')

        optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

        loss_per_epoch = []
        R_squared_per_epoch = []

        n_epochs = 50

        start_time = time.time()
        # Run the training loop
        for epoch in range(0, n_epochs): # n_epochs at maximum

            # Print epoch
            print(f'Starting epoch {epoch+1}')

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
                
                error = torch.abs(outputs - targets)
                # Compute R^2 score
                ss_residual = torch.sum((targets - outputs)**2)
                ss_total = torch.sum((targets - torch.mean(targets))**2)
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
            
            R_squared_per_epoch.append(np.mean(R_squared_per_batch))
            print(f'\tAfter epoch {epoch+1}: Loss = {loss_per_epoch[epoch]}, R-squared = {R_squared_per_epoch[epoch]}')
            
        print('Training process has finished.')

        end_time = time.time()
        print(f'Training time: {end_time - start_time} seconds\n')

        fig, axs = plt.subplots(2,1,figsize=(8,8))

        axs[0].plot(np.arange(1,len(loss_per_epoch)+1), loss_per_epoch, color='blue', label='Training loss', marker='.')
        axs[0].grid(True)
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_ylim([0, 1])
        axs[0].legend() 

        axs[1].plot(np.arange(1,len(R_squared_per_epoch)+1), R_squared_per_epoch, color='blue', label='Training R squared', marker='x')
        axs[1].grid(True)
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('R squared')
        axs[1].set_ylim([0, 1])
        axs[1].legend()
        plt.suptitle('Loss and training R squared curves during training', y=0.92)
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
        test_predictions = test_predictions.reshape(-1)
        test_targets = np.array(dataset_test.y) * std_permeability + mean_permeability
        # all_predictions = outputs_all.cpu().numpy()
        all_predictions = all_predictions * std_permeability + mean_permeability
        all_predictions = all_predictions.reshape(-1)
        all_targets = permeability_values

        # compute the test R squared
        ss_residual = np.sum((test_targets - test_predictions)**2)
        ss_total = np.sum((test_targets - np.mean(test_targets))**2)
        R_squared_test = 1 - (ss_residual / ss_total)
        
        # Ensure all_targets and all_predictions have the same length
        if len(all_targets) != len(all_predictions):
            raise ValueError("Length of all_targets and all_predictions must be the same")

        # Compute the residual sum of squares and total sum of squares
        ss_residual = np.sum((all_targets - all_predictions)**2)
        ss_total = np.sum((all_targets - np.mean(all_targets))**2)
        R_squared_all = 1 - (ss_residual / ss_total)
    

        # Visualize the ground truth on x axis and predicted values on y axis
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(all_targets, all_predictions, color='blue', label='Predictions')
        ax.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'k--', lw=2, label='Ideal')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title('Ground Truth vs Predicted Values')
        plt.xscale('log')
        plt.yscale('log')
        ax.legend()
        ax.text(0.05, 0.95, f'R^2: {R_squared_all:.2f}', transform=ax.transAxes, fontsize=14, verticalalignment='top')
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Log_results/Truth_vs_Predicted-all-logarithmic-{cnn.__class__.__name__}-{lr}.png'))

        # Visualize the ground truth on x axis and predicted values on y axis
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(test_targets, test_predictions, color='blue', label='Predictions')
        ax.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'k--', lw=2, label='Ideal')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title('Ground Truth vs Predicted Values')
        plt.xscale('log')
        plt.yscale('log')
        ax.legend()
        ax.text(0.05, 0.95, f'Test R^2: {R_squared_test:.2f}', transform=ax.transAxes, fontsize=14, verticalalignment='top')
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Log_results/Truth_vs_Predicted-test-logarithmic-{cnn.__class__.__name__}-{lr}.png'))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(all_targets, all_predictions, color='blue', label='Predictions')
        ax.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'k--', lw=2, label='Ideal')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title('Ground Truth vs Predicted Values')
        ax.legend()
        ax.text(0.05, 0.95, f'R^2: {R_squared_all:.2f}', transform=ax.transAxes, fontsize=14, verticalalignment='top')
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Normal_results/Truth_vs_Predicted-all-{cnn.__class__.__name__}-{lr}.png'))
        

        # Visualize the ground truth on x axis and predicted values on y axis
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(test_targets, test_predictions, color='blue', label='Predictions')
        ax.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'k--', lw=2, label='Ideal')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title('Ground Truth vs Predicted Values')
        ax.legend()
        ax.text(0.05, 0.95, f'Test R^2: {R_squared_test:.2f}', transform=ax.transAxes, fontsize=14, verticalalignment='top')
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Normal_results/Truth_vs_Predicted-test-{cnn.__class__.__name__}-{lr}.png'))
        plt.close('all')

        # Free up memory to avoid 'CUDA out of memory' error when moving from one iteration to the next
        del test_inputs
        del data_images_np
        del data_images_tensor
        del test_predictions
        del test_targets
        del all_predictions
        del all_targets
        del R_squared_all
        del R_squared_test
        torch.cuda.empty_cache()
        cnn = cnn.to(my_device) 