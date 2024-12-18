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
from classes_cnn_conv import NoPoolCNN1
import time

# Default dype is float64. Not working currently on DelftBlue
torch.set_default_dtype(torch.float64)

# Set fixed random number seed for reproducibility of random initializations
torch.manual_seed(42)

# Choose whether to use take account 15% validation set or not. Not used at the moment
validation = True
image_size = 400

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
        df['Packing_Fraction'] = packing_fraction
        df['Shape'] = shape
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
# Full_1000
# Quarter_1000
# Full_2000


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
            # image = image[:, :, :3]  # Ignore the alpha channel
            image = image[:, :, 0]  # Take only the first channel
        
        data_images.append({
            'Model': model_number,
            # 'Packing_Fraction': packing_fraction,
            # 'Shape': shape,
            'Image': image
        })
        # Delete images that have a model number = 1
        if model_number == 1:
            data_images.pop()
            continue

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
batch_size = 32
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

loss_function = nn.MSELoss()

def reset_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):  # Include other layers if needed
        m.reset_parameters()

for cnn in [NoPoolCNN1().to(my_device)]:
    start_time = time.time()
    for lr in [1e-3]:
        cnn.apply(reset_weights)
        print(f'Using CNN: {cnn.__class__.__name__} with learning rate: {lr}')

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

            # Log loss value per epoch
            epoch_loss = np.mean(loss_per_batch)
            loss_per_epoch.append(epoch_loss)
            R_squared_per_epoch.append(np.mean(R_squared_per_batch))
            print(f'\tAfter epoch {epoch+1}: Loss = {loss_per_epoch[epoch]}, R-squared = {R_squared_per_epoch[epoch]}')

            if validation:
                val_loss_per_batch = []
                val_R_squared_per_batch = []
                with torch.no_grad():
                    for val_data in torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0):
                        val_inputs, val_targets = val_data
                        val_inputs = val_inputs.to(my_device)
                        val_targets = val_targets.to(my_device)

                        val_outputs = cnn(val_inputs)
                        val_outputs = val_outputs.squeeze()

                        val_loss = loss_function(val_outputs, val_targets)
                        val_ss_residual = torch.sum((val_targets - val_outputs)**2)
                        val_ss_total = torch.sum((val_targets - torch.mean(val_targets))**2)
                        val_R_squared = 1 - (val_ss_residual / val_ss_total)

                        val_loss_per_batch.append(val_loss.item())
                        val_R_squared_per_batch.append(val_R_squared.item())

                val_epoch_loss = np.mean(val_loss_per_batch)
                val_loss_per_epoch.append(val_epoch_loss)
                val_R_squared_per_epoch.append(np.mean(val_R_squared_per_batch))
                print(f'\tValidation Loss = {val_loss_per_epoch[epoch]}, Validation R-squared = {val_R_squared_per_epoch[epoch]}')

            # Save the best model based on validation loss if validation is used, otherwise based on training loss
            if validation:
                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    best_model = cnn.state_dict()
            else:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model = cnn.state_dict()
            
        print('Training process has finished.')

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
        plt.suptitle('Loss and training R squared curves during training', y=0.92)
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Temp-{image_size}-Evolution'), exist_ok=True)
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Temp-{image_size}-Evolution/Loss_R_squared-{cnn.__class__.__name__}-{lr}.png'))

        # Evaluate model and clear cache
        cnn.load_state_dict(best_model)
        cnn.eval()
        torch.cuda.empty_cache()

        with torch.no_grad():
            test_inputs = dataset_test.X.to(my_device)
            test_predictions = cnn(test_inputs).cpu().numpy()
            # all_inputs = torch.cat((dataset_train.X.to(my_device), dataset_test.X.to(my_device)), dim=0)
            # all_predictions = cnn(all_inputs).cpu().numpy()
            # data_images_np = np.array(data_images)
            # data_images_tensor = torch.tensor(data_images_np).unsqueeze(1).to('cpu')
            # all_predictions = cnn(data_images_tensor).cpu().numpy()

        test_predictions = test_predictions * std_permeability + mean_permeability
        test_predictions = test_predictions.reshape(-1)
        test_targets = np.array(dataset_test.y) * std_permeability + mean_permeability
        # all_predictions = all_predictions * std_permeability + mean_permeability
        # all_predictions = all_predictions.reshape(-1)
        # all_targets = permeability_values

        # compute the test R squared
        ss_residual = np.sum((test_targets - test_predictions)**2)
        ss_total = np.sum((test_targets - np.mean(test_targets))**2)
        R_squared_test = 1 - (ss_residual / ss_total)
        
        
        # Visualize the ground truth on x axis and predicted values on y axis
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(test_targets, test_predictions, color='blue', label='Predictions')
        ax.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], 'k--', lw=2, label='Ideal')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title('Ground Truth vs Predicted Values')
        plt.xscale('log')
        plt.yscale('log')
        ax.legend(loc='lower right')
        ax.text(0.05, 0.95, f'Test R^2: {R_squared_test:.5f}', transform=ax.transAxes, fontsize=14, verticalalignment='top')
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Temp-{image_size}-Log_results'), exist_ok=True)
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Temp-{image_size}-Log_results/Truth_vs_Predicted-test-logarithmic-{cnn.__class__.__name__}-{lr}.png'))
        
        # Visualize the ground truth on x axis and predicted values on y axis
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(test_targets, test_predictions, color='blue', label='Predictions')
        ax.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], 'k--', lw=2, label='Ideal')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title('Ground Truth vs Predicted Values')
        ax.legend(loc='lower right')
        ax.text(0.05, 0.95, f'Test R^2: {R_squared_test:.5f}', transform=ax.transAxes, fontsize=14, verticalalignment='top')
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Temp-{image_size}-Normal_results'), exist_ok=True)
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), f'Temp-{image_size}-Normal_results/Truth_vs_Predicted-test-{cnn.__class__.__name__}-{lr}.png'))
        plt.close('all')

        # Free up memory to avoid 'CUDA out of memory' error when moving from one iteration to the next
        #cnn.to(my_device)
        del test_inputs
        del test_predictions
        del test_targets
        del R_squared_test
        torch.cuda.empty_cache()


    time_end = time.time()
    print(f'Time taken for {cnn.__class__.__name__}: {time_end - start_time} seconds')
