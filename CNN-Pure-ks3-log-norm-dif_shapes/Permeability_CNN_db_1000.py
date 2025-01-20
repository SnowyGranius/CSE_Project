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
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, r2_score

# Default dype is float64
torch.set_default_dtype(torch.float64)

# Set fixed random number seed for reproducibility of random initializations
torch.manual_seed(42)

# Choose whether to use take account 15% validation set or not. Not used at the moment
validation = True
image_size = 1000
training_shapes = ['circle']

# Use CUDA
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {my_device}')

# Load the data from csv files and read the images
csv_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'Porespy_homogenous_diamater')
if not os.path.exists(csv_directory):
    raise FileNotFoundError(f"Directory {csv_directory} does not exist.")

data_csv = []
data_csv_training = []
data_images = []
data_images_training = []
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
        df['Shape'] = shape
        # # Delete the first row of the dataframe without using pandas
        for key in df.keys():
            df[key] = df[key][1:]
        if shape in training_shapes:
            data_csv_training.append(df)
        else:
            data_csv.append(df)

permeability_values = []
for df in data_csv:
    permeability_values.extend(df['Permeability'])

permeability_values_training = []
for df in data_csv_training:
    permeability_values_training.extend(df['Permeability'])

print(f'Number of permeability values: {len(permeability_values)}')

permeability_values = np.array(permeability_values)
permeability_values_training = np.array(permeability_values_training)


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
        
        if model_number == 1:
            continue
        if shape in training_shapes:
            data_images_training.append({'Image': image})
        else:
            data_images.append({'Image': image})

# Transform the images to numpy arrays containing 0's and 1's
data_images = [np.array(image['Image'], dtype=np.float64) for image in data_images]
data_images_training = [np.array(image['Image'], dtype=np.float64) for image in data_images_training]
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
    test_images, val_images, test_permeability, val_permeability = train_test_split(
        data_images, permeability_values, test_size=0.50, random_state=42)

train_permeability = permeability_values_training
train_images = data_images_training


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


# Initialize the dataloader using batch size hyperparameter
batch_size = 32
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

# Define the loss function
loss_function = nn.MSELoss()

# Function that resets the weights of the model at each iteration (when running on DelftBlue, 
    # the model seems to be memorized from one learning rate to the next and gives optimistic results)
def reset_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):  # Include other layers if needed
        m.reset_parameters()

# Training loop
for cnn in [NoPoolCNN1().to(my_device)]:
    start_time = time.time()
    lr_list = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    for lr in lr_list:
        cnn.apply(reset_weights)
        print(f'Using CNN: {cnn.__class__.__name__} with learning rate: {lr:.0e}')

        # Define the optimizer
        optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

        loss_per_epoch = []
        R_squared_per_epoch = []
        val_loss_per_epoch = []
        val_R_squared_per_epoch = []

        n_epochs = 50

        # Run the training loop
        best_loss = float('inf')
        best_model = None
        for epoch in range(0, n_epochs): # n_epochs at maximum

            # Print epoch
            #print(f'Starting epoch {epoch+1}')

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
                

            # Training loss and training R squared per epoch
            epoch_loss = np.mean(loss_per_batch)
            loss_per_epoch.append(epoch_loss)
            R_squared_per_epoch.append(np.mean(R_squared_per_batch))
            print(f'After epoch {epoch+1}: Loss = {loss_per_epoch[epoch]}, R-squared = {R_squared_per_epoch[epoch]}')
            
            # Validation loss and validation R squared per epoch - this is preferred over loss and R squared on the training set
            if validation:
                val_loss_per_batch = []
                val_R_squared_per_batch = []
                cnn.eval()
                with torch.no_grad():
                    for val_data in torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0):
                        # Get and prepare inputs by converting to floats
                        val_inputs, val_targets = val_data
                        val_inputs = val_inputs.to(my_device)
                        val_targets = val_targets.to(my_device)

                        # Perform forward pass
                        val_outputs = cnn(val_inputs)
                        val_outputs = val_outputs.squeeze()

                        # Compute loss and R squared
                        val_loss = loss_function(val_outputs, val_targets)
                        val_ss_residual = torch.sum((val_targets - val_outputs)**2)
                        val_ss_total = torch.sum((val_targets - torch.mean(val_targets))**2)
                        val_R_squared = 1 - (val_ss_residual / val_ss_total)

                        val_loss_per_batch.append(val_loss.item())
                        val_R_squared_per_batch.append(val_R_squared.item())

                # Validation loss and validation R squared per epoch
                val_epoch_loss = np.mean(val_loss_per_batch)
                val_loss_per_epoch.append(val_epoch_loss)
                val_R_squared_per_epoch.append(np.mean(val_R_squared_per_batch))
                print(f'\tValidation Loss = {val_loss_per_epoch[epoch]}, Validation R-squared = {val_R_squared_per_epoch[epoch]}')

            # Save the best model based on validation loss if validation is used, otherwise based on training loss.
                # If validation is not used, the model is saved based on the training loss - not recommended
            if validation:
                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    best_model = cnn.state_dict()
            else:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model = cnn.state_dict()
            cnn.train()
            
        print('Training process has finished.\n')

        # Plot the loss and R squared curves on training and validation sets
        fig, axs = plt.subplots(2,1,figsize=(8,8))

        axs[0].plot(np.arange(1,len(loss_per_epoch)+1), loss_per_epoch, color='blue', label='Training loss', marker='.')
        if validation:
            axs[0].plot(np.arange(1,len(val_loss_per_epoch)+1), val_loss_per_epoch, color='red', label='Validation loss', marker='.')
        axs[0].grid(True)
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_ylim([0, 1])
        axs[0].legend(loc='lower right')

        axs[1].plot(np.arange(1,len(R_squared_per_epoch)+1), R_squared_per_epoch, color='blue', label='Training R squared', marker='x')
        if validation:
            axs[1].plot(np.arange(1,len(val_R_squared_per_epoch)+1), val_R_squared_per_epoch, color='red', label='Validation R squared', marker='x')
        axs[1].grid(True)
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('R squared')
        axs[1].set_ylim([0, 1])
        axs[1].legend(loc='lower right')
        plt.suptitle(f'Image size {image_size}, {cnn.__class__.__name__}, lr = {lr:.0e}', y=0.92)
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Evolution'), exist_ok=True)
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Evolution/Loss_R_squared-{cnn.__class__.__name__}-{lr:.0e}.png'))

        # Save the best model to a file in the current folder
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Models'), exist_ok=True)
        torch.save(cnn.state_dict(), os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Models/best_model-{cnn.__class__.__name__}-{lr:.0e}.pth'))

        # Evaluate model and clear cache
        cnn.load_state_dict(best_model)
        cnn.eval()
        torch.cuda.empty_cache()

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
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Log_results/Truth_vs_Predicted-test-logarithmic-{cnn.__class__.__name__}-{lr:.0e}.png'))
        
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
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Normal_results/Truth_vs_Predicted-test-{cnn.__class__.__name__}-{lr:.0e}.png'))
        plt.close('all')

        # Free up memory to avoid 'CUDA out of memory' error when moving from one iteration to the next
        # cnn.to(my_device)
        del test_inputs
        del test_predictions
        del test_targets
        del R_squared_test
        torch.cuda.empty_cache()


    time_end = time.time()
    print(f'Time taken for {cnn.__class__.__name__}, {len(lr_list)} learning rates: {time_end - start_time} seconds\n')