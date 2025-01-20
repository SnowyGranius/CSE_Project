import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import os
import sys
import glob
import re
from sklearn.model_selection import train_test_split
from classes_cnn_1000 import NoPoolCNN1
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Default dype is float64
torch.set_default_dtype(torch.float64)

# Set fixed random number seed for reproducibility of random initializations
torch.manual_seed(42)

# Choose whether to use take account 15% validation set or not. Not used at the moment
validation = True
image_size = 1000

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
all_csv.sort()

# Reading all entries (circles, ellipses, triangles, rectangles) from the csv files, except model 1 which gives eronous results
for file in all_csv:
    if 'circle' in file or 'ellipse' in file or 'triangle' in file or 'rectangle' in file:
        with open(file, 'r') as f:
            lines = f.readlines()
            headers = lines[0].strip().split(',')
            data = [line.strip().split(',') for line in lines[1:]]
            df = {header: [] for header in headers}
            for row in data:
                for header, value in zip(headers, row):
                    df[header].append(float(value) if header == 'Permeability' else value)
        packing_fraction = re.search(r'\d.\d+', file).group()
        shape = re.search(r'circle|ellipse|triangle|rectangle', file).group()
        # df['Packing_Fraction'] = packing_fraction
        # df['Shape'] = shape
        df['Model'] = list(range(1, len(df['Permeability'])+1))  # Model number is one higher than the index of the dataframe
        # Delete the first row of the dataframe without using pandas
        for key in df.keys():
            df[key] = df[key][1:]
        data_csv.append(df)

permeability_values = []
for df in data_csv:
    permeability_values.extend(df['Permeability'])

print(f'Number of permeability values: {len(permeability_values)}')

permeability_values = np.array(permeability_values)

# Read images from the folder
image_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), f'Image_dataset_generation/Converge-study-{image_size}')
if not os.path.exists(image_directory):
    raise FileNotFoundError(f"Directory {image_directory} does not exist.")

all_images = glob.glob(os.path.join(image_directory, "*.png"))
all_images.sort()

for image_file in all_images:
    match = re.search(r'pf_(\d\.\d+)_(circle|ellipse|triangle|rectangle)_Model_(\d+)\.png', image_file)
    if match:
        packing_fraction = float(match.group(1))
        shape = match.group(2)
        model_number = int(match.group(3))
        image = plt.imread(image_file)
        if image.ndim == 3:
            image = image[:, :, 0]  # Take only the first channel
        
        data_images.append({
            'Model': model_number,
            'Image': image
        })
        # Delete images that have a model number = 1
        if model_number == 1:
            data_images.pop()
            continue

# Transform the images to numpy arrays containing 0's and 1's
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
scaler_permeability = StandardScaler()
train_permeability = np.log(train_permeability)
test_permeability = np.log(test_permeability)
train_permeability = scaler_permeability.fit_transform(train_permeability.reshape(-1, 1)).flatten()
test_permeability = scaler_permeability.transform(test_permeability.reshape(-1, 1)).flatten()
dataset_train = PermeabilityDataset(X=train_images, y=train_permeability)
dataset_test = PermeabilityDataset(X=test_images, y=test_permeability)

if validation:
    val_permeability = np.log(val_permeability)
    val_permeability = scaler_permeability.transform(val_permeability.reshape(-1, 1)).flatten()
    dataset_val = PermeabilityDataset(X=val_images, y=val_permeability)

for cnn in [NoPoolCNN1().to(my_device)]:
    for lr in [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
        print(f'Model: {cnn.__class__.__name__}, Learning Rate: {lr:.0e}')
        cnn.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Models/best_model-{cnn.__class__.__name__}-{lr:.0e}.pth'), weights_only=True))
        cnn.eval()

        # Evaluate the model on the test set
        with torch.no_grad():
            test_inputs = dataset_test.X.to(my_device)
            test_predictions = cnn(test_inputs).cpu().numpy()
            test_targets = np.array(dataset_test.y)

        # Inverse transform the permeability values to their original scale
        test_predictions = scaler_permeability.inverse_transform(test_predictions.reshape(-1, 1)).reshape(-1)
        test_targets = scaler_permeability.inverse_transform(test_targets.reshape(-1, 1)).reshape(-1)
        test_predictions = np.exp(test_predictions)
        test_targets = np.exp(test_targets)
        R_squared_test = r2_score(test_targets, test_predictions)
        mse_loss_test = mean_squared_error(test_targets, test_predictions)
        # print(test_targets)
        # print(test_predictions)
        # Visualize the ground truth on x axis and predicted values on y axis - logarithmic scale
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(test_targets, test_predictions, alpha=0.6, color='blue', label=f'R²: {R_squared_test:.5f}\nMSE: {mse_loss_test:.5g}')
        ax.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], color='r', linestyle='--', label='Parity Line')
        ax.set_xlabel('True Permeability')
        ax.set_ylabel('Predicted Permeability')
        ax.set_title(f'Parity Plot Pure CNN, Image Size {image_size}x{image_size}, {cnn.__class__.__name__}, lr = {lr:.0e}')
        plt.xscale('log')
        plt.yscale('log')
        ax.legend(loc='lower right', fontsize='large')
        ax.set_aspect('equal', 'box')
        ax.grid(True)
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Log_results'), exist_ok=True)
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Log_results/Truth_vs_Predicted-test-logarithmic-{cnn.__class__.__name__}-{lr:.0e}.png'))
        
        # Visualize the ground truth on x axis and predicted values on y axis - normal scale
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(test_targets, test_predictions, alpha=0.6, color='blue', label=f'R²: {R_squared_test:.5f}\nMSE: {mse_loss_test:.5g}')
        ax.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], color='r', linestyle='--', label='Parity Line')
        ax.set_xlabel('True Permeability')
        ax.set_ylabel('Predicted Permeability')
        ax.set_title(f'Parity Plot Pure CNN, Image Size {image_size}x{image_size}, {cnn.__class__.__name__}, lr = {lr:.0e}')
        ax.legend(loc='lower right', fontsize='large')
        ax.set_aspect('equal', 'box')
        ax.grid(True)
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Normal_results'), exist_ok=True)
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Normal_results/Truth_vs_Predicted-test-{cnn.__class__.__name__}-{lr:.0e}.png'))
        plt.close('all')

        # Free up memory to avoid 'CUDA out of memory' error when moving from one iteration to the next
        # cnn.to(my_device)
        del test_inputs
        del test_predictions
        del test_targets
        torch.cuda.empty_cache()