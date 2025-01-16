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
from classes_cnn_informed_800 import NoPoolCNN11, NoPoolCNN12, NoPoolCNN13, NoPoolCNN14, NoPoolCNN15, NoPoolCNN16
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Default dype is float64
torch.set_default_dtype(torch.float64)

# Set fixed random number seed for reproducibility of random initializations
torch.manual_seed(42)

# Choose whether to use take account 15% validation set or not. Not used at the moment
validation = True
image_size = 800

# Use CUDA
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {my_device}')


####### blobs reading here #######
# Load the data from csv files and read the images
csv_directory = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'blobs_cnn')
if not os.path.exists(csv_directory):
    raise FileNotFoundError(f"Directory {csv_directory} does not exist.")

data_csv_blobs = []
data_images_blobs = []
all_csv_blobs = glob.glob(os.path.join(csv_directory, "*.csv"))
all_csv_blobs.sort()

# Reading all entries (circles, ellipses, triangles, rectangles) from the csv files, except model 1 which gives eronous results
for file in all_csv_blobs:
    if 'blobs' in file:
        with open(file, 'r') as f:
            lines = f.readlines()
            headers = lines[0].strip().split(',')
            data = [line.strip().split(',') for line in lines[1:]]
            df = {header: [] for header in headers}
            for row in data:
                for header, value in zip(headers, row):
                    if header in ['Permeability_mean', 'Porosity_mean', 'Surface_mean', 'Euler_total']:
                        df[header].append(float(value))
        data_csv_blobs.append(df)

permeability_values_blobs = []
porosity_values_blobs = []
surface_values_blobs = []
euler_total_values_blobs = []
for df in data_csv_blobs:
    permeability_values_blobs.extend(df['Permeability_mean'])
    porosity_values_blobs.extend(df['Porosity_mean'])
    surface_values_blobs.extend(df['Surface_mean'])
    euler_total_values_blobs.extend(df['Euler_total'])

print(f'Number of permeability values: {len(permeability_values_blobs)}')

permeability_values_blobs = np.array(permeability_values_blobs)
porosity_values_blobs = np.array(porosity_values_blobs)
surface_values_blobs = np.array(surface_values_blobs)
euler_total_values_blobs = np.array(euler_total_values_blobs)

minkowski_values_blobs = np.column_stack((porosity_values_blobs, surface_values_blobs, euler_total_values_blobs))
permeability_minkowski_blobs = np.column_stack((permeability_values_blobs, minkowski_values_blobs))

# Read images from the folder
image_directory = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'blobs_cnn')
if not os.path.exists(image_directory):
    raise FileNotFoundError(f"Directory {image_directory} does not exist.")

all_images_blobs = glob.glob(os.path.join(image_directory, "*.png"))
all_images_blobs.sort()

for image_file in all_images_blobs:
    image = plt.imread(image_file)
    if image.ndim == 3:
        image = image[:, :, 0]  # Take only the first channel
    # invert the image
    image = 1 - image
    data_images_blobs.append({
        'Image': image
    })

# Transform the images to numpy arrays containing 0's and 1's
data_images_blobs = [np.array(image['Image'], dtype=np.float64) for image in data_images_blobs]
print(f'Number of images: {len(data_images_blobs)}')

####### reading regular dataset here #######
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
                    if header in ['Permeability', 'Porosity', 'Surface', 'Euler_total']:
                        df[header].append(float(value))
        packing_fraction = re.search(r'\d.\d+', file).group()
        shape = re.search(r'circle|ellipse|triangle|rectangle', file).group()
        for key in df.keys():
            df[key] = df[key][1:]
        data_csv.append(df)

permeability_values = []
porosity_values = []
surface_values = []
euler_total_values = []
for df in data_csv:
    permeability_values.extend(df['Permeability'])
    porosity_values.extend(df['Porosity'])
    surface_values.extend(df['Surface'])
    euler_total_values.extend(df['Euler_total'])

print(f'Number of permeability values: {len(permeability_values)}')

permeability_values = np.array(permeability_values)
porosity_values = np.array(porosity_values)
surface_values = np.array(surface_values)
euler_total_values = np.array(euler_total_values)

minkowski_values = np.column_stack((porosity_values, surface_values, euler_total_values))
permeability_minkowski = np.column_stack((permeability_values, minkowski_values))

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
            image = image[:, :, 0]  # Take only the first channel (equivalent to grayscale)
        
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

# Define the dataset class that converts to torch tensors
class PermeabilityDataset(torch.utils.data.Dataset):
    '''
    Permeability dataset class for regression, with Minkowski variables.
    '''
    def __init__(self, X, minkowski, y):
        if not torch.is_tensor(X):
            X = np.expand_dims(X, axis=1)
            self.X = torch.tensor(X)
        if not torch.is_tensor(minkowski):
            self.minkowski = torch.tensor(minkowski)
        if not torch.is_tensor(y):
            self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.minkowski[i], self.y[i]



# Splitting the data into training, testing (and validation) sets
if validation:
    train_images, dummy_images, train_permeability_minkowski, dummy_permeability_minkowski = train_test_split(
        data_images, permeability_minkowski, test_size=0.30, random_state=42)

    test_images, val_images, test_permeability_minkowski, val_permeability_minkowski = train_test_split(
        dummy_images, dummy_permeability_minkowski, test_size=0.50, random_state=42)

else:
    train_images, test_images, train_permeability_minkowski, test_permeability_minkowski = train_test_split(
        data_images, permeability_minkowski, test_size=0.30, random_state=42)

test_permeability_minkowski = permeability_minkowski_blobs
test_images = data_images_blobs

# Normalizing the training permeability and Minkowski values using StandardScaler()
scaler_permeability = StandardScaler()
scaler_porosity = StandardScaler()
scaler_surface = StandardScaler()
scaler_euler_total = StandardScaler()

# Apply the logaritmic transformation to the permeability values
train_permeability_minkowski[:, 0] = np.log(train_permeability_minkowski[:, 0])
train_permeability = scaler_permeability.fit_transform(train_permeability_minkowski[:, 0].reshape(-1, 1)).reshape(-1)
train_porosity = scaler_porosity.fit_transform(train_permeability_minkowski[:, 1].reshape(-1, 1)).reshape(-1)
train_surface = scaler_surface.fit_transform(train_permeability_minkowski[:, 2].reshape(-1, 1)).reshape(-1)
train_euler_total = scaler_euler_total.fit_transform(train_permeability_minkowski[:, 3].reshape(-1, 1)).reshape(-1)

# Normalizing the testing and validation permeability and minkowski values using StandardScaler()
test_permeability_minkowski[:, 0] = np.log(test_permeability_minkowski[:, 0])
test_permeability = scaler_permeability.transform(test_permeability_minkowski[:, 0].reshape(-1, 1)).reshape(-1)
test_porosity = scaler_porosity.transform(test_permeability_minkowski[:, 1].reshape(-1, 1)).reshape(-1)
test_surface = scaler_surface.transform(test_permeability_minkowski[:, 2].reshape(-1, 1)).reshape(-1)
test_euler_total = scaler_euler_total.transform(test_permeability_minkowski[:, 3].reshape(-1, 1)).reshape(-1)

train_minkowski = np.column_stack((train_porosity, train_surface, train_euler_total))
test_minkowski = np.column_stack((test_porosity, test_surface, test_euler_total))

dataset_train = PermeabilityDataset(X=train_images, minkowski=train_minkowski, y=train_permeability)
dataset_test = PermeabilityDataset(X=test_images, minkowski=test_minkowski, y=test_permeability)

if validation:
    val_permeability_minkowski[:, 0] = np.log(val_permeability_minkowski[:, 0])
    val_permeability = scaler_permeability.transform(val_permeability_minkowski[:, 0].reshape(-1, 1)).reshape(-1)
    val_porosity = scaler_porosity.transform(val_permeability_minkowski[:, 1].reshape(-1, 1)).reshape(-1)
    val_surface = scaler_surface.transform(val_permeability_minkowski[:, 2].reshape(-1, 1)).reshape(-1)
    val_euler_total = scaler_euler_total.transform(val_permeability_minkowski[:, 3].reshape(-1, 1)).reshape(-1)

    val_minkowski = np.column_stack((val_porosity, val_surface, val_euler_total))

    dataset_val = PermeabilityDataset(X=val_images, minkowski=val_minkowski, y=val_permeability)

for cnn in [NoPoolCNN11().to(my_device), NoPoolCNN12().to(my_device), NoPoolCNN13().to(my_device), NoPoolCNN14().to(my_device), NoPoolCNN15().to(my_device), NoPoolCNN16().to(my_device)]:
    for lr in [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
        print(f'Model: {cnn}, Learning Rate: {lr:.0e}')
        cnn.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Models/best_model-{cnn.__class__.__name__}-{lr:.0e}.pth')))
        cnn.eval()

        # Evaluate the model on the test set
        with torch.no_grad():
            test_inputs = dataset_test.X.to(my_device)
            test_minkowski = dataset_test.minkowski.to(my_device)
            test_predictions = cnn(test_inputs, test_minkowski).cpu().numpy()
            test_targets = np.array(dataset_test.y)

        # Inverse transform the permeability values to their original scale and calculate R squared and MSE loss
        test_predictions = scaler_permeability.inverse_transform(test_predictions.reshape(-1, 1)).reshape(-1)
        test_targets = scaler_permeability.inverse_transform(test_targets.reshape(-1, 1)).reshape(-1)
        test_predictions = np.exp(test_predictions)
        test_targets = np.exp(test_targets)
        R_squared_test = r2_score(test_targets, test_predictions)
        mse_loss_test = mean_squared_error(test_targets, test_predictions)

        # Visualize the ground truth on x axis and predicted values on y axis - logarithmic scale
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(test_targets, test_predictions, alpha=0.6, color='blue', label=f'R²: {R_squared_test:.5f}\nMSE: {mse_loss_test:.5g}')
        ax.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], color='r', linestyle='--', label='Parity Line')
        ax.set_xlabel('True Permeability')
        ax.set_ylabel('Predicted Permeability')
        ax.set_title(f'Parity Plot Informed CNN, Image Size {image_size}x{image_size}, {cnn.__class__.__name__}, lr = {lr:.0e}')
        plt.xscale('log')
        plt.yscale('log')
        ax.legend(loc='lower right', fontsize='large')
        ax.set_aspect('equal', 'box')
        ax.grid(True)
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Log_results'), exist_ok=True)
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Log_results/Truth_vs_Predicted-test-logarithmic-{cnn.__class__.__name__}-{lr:.0e}-blobs.png'))
        
        # Visualize the ground truth on x axis and predicted values on y axis - normal scale
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(test_targets, test_predictions, alpha=0.6, color='blue', label=f'R²: {R_squared_test:.5f}\nMSE: {mse_loss_test:.5g}')
        ax.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], color='r', linestyle='--', label='Parity Line')
        ax.set_xlabel('True Permeability')
        ax.set_ylabel('Predicted Permeability')
        ax.set_title(f'Parity Plot Informed CNN, Image Size {image_size}x{image_size}, {cnn.__class__.__name__}, lr = {lr:.0e}')
        ax.legend(loc='lower right', fontsize='large')
        ax.set_aspect('equal', 'box')
        ax.grid(True)
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Normal_results'), exist_ok=True)
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Normal_results/Truth_vs_Predicted-test-{cnn.__class__.__name__}-{lr:.0e}-blobs.png'))
        plt.close('all')

        # Free up memory to avoid 'CUDA out of memory' error when moving from one iteration to the next
        # cnn.to(my_device)
        del test_inputs
        del test_predictions
        del test_targets
        del R_squared_test
        torch.cuda.empty_cache()