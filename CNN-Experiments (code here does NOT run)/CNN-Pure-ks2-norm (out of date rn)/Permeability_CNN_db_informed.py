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
from classes_cnn_informed import NoPoolCNN11, NoPoolCNN12, NoPoolCNN13
import time
from sklearn.preprocessing import StandardScaler

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
                    if header in ['Permeability', 'Porosity', 'Surface', 'Euler_total']:
                        df[header].append(float(value))
        packing_fraction = re.search(r'\d.\d+', file).group()
        shape = re.search(r'circle|ellipse|triangle|rectangle', file).group()
        # df['Packing_Fraction'] = packing_fraction
        # df['Shape'] = shape
        # df['Model'] = list(range(1, len(df['Permeability'])+1))  # Model number is one higher than the index of the dataframe
        # # Delete the first row of the dataframe without using pandas
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
# Converge-study-{image_size}
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


# Normalizing the training permeability and minkowski values using StandardScaler()
scaler_permeability = StandardScaler()
scaler_porosity = StandardScaler()
scaler_surface = StandardScaler()
scaler_euler_total = StandardScaler()

train_permeability = scaler_permeability.fit_transform(train_permeability_minkowski[:, 0].reshape(-1, 1)).reshape(-1)
train_porosity = scaler_porosity.fit_transform(train_permeability_minkowski[:, 1].reshape(-1, 1)).reshape(-1)
train_surface = scaler_surface.fit_transform(train_permeability_minkowski[:, 2].reshape(-1, 1)).reshape(-1)
train_euler_total = scaler_euler_total.fit_transform(train_permeability_minkowski[:, 3].reshape(-1, 1)).reshape(-1)

# Normalizing the testing and validation permeability and minkowski values using StandardScaler()
test_permeability = scaler_permeability.transform(test_permeability_minkowski[:, 0].reshape(-1, 1)).reshape(-1)
test_porosity = scaler_porosity.transform(test_permeability_minkowski[:, 1].reshape(-1, 1)).reshape(-1)
test_surface = scaler_surface.transform(test_permeability_minkowski[:, 2].reshape(-1, 1)).reshape(-1)
test_euler_total = scaler_euler_total.transform(test_permeability_minkowski[:, 3].reshape(-1, 1)).reshape(-1)

train_minkowski = np.column_stack((train_porosity, train_surface, train_euler_total))
test_minkowski = np.column_stack((test_porosity, test_surface, test_euler_total))

dataset_train = PermeabilityDataset(X=train_images, minkowski=train_minkowski, y=train_permeability)
dataset_test = PermeabilityDataset(X=test_images, minkowski=test_minkowski, y=test_permeability)

if validation:
    val_permeability = scaler_permeability.transform(val_permeability_minkowski[:, 0].reshape(-1, 1)).reshape(-1)
    val_porosity = scaler_porosity.transform(val_permeability_minkowski[:, 1].reshape(-1, 1)).reshape(-1)
    val_surface = scaler_surface.transform(val_permeability_minkowski[:, 2].reshape(-1, 1)).reshape(-1)
    val_euler_total = scaler_euler_total.transform(val_permeability_minkowski[:, 3].reshape(-1, 1)).reshape(-1)

    val_minkowski = np.column_stack((val_porosity, val_surface, val_euler_total))

    dataset_val = PermeabilityDataset(X=val_images, minkowski=val_minkowski, y=val_permeability)


# Initialize the dataloader using batch size hyperparameter
batch_size = 32
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

loss_function = nn.MSELoss()

def reset_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):  # Include other layers if needed
        m.reset_parameters()

for cnn in [NoPoolCNN11().to(my_device), NoPoolCNN12().to(my_device), NoPoolCNN13().to(my_device)]:
    start_time = time.time()
    # for lr in [5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:
    for lr in [1e-3]:
        cnn.apply(reset_weights)
        print(f'Using CNN: {cnn.__class__.__name__} with learning rate: {lr}')

        optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

        loss_per_epoch = []
        R_squared_per_epoch = []
        val_loss_per_epoch = []
        val_R_squared_per_epoch = []

        n_epochs = 1

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
                inputs, minkowski, targets = data
                # Transfer data to your chosen device
                inputs = inputs.to(my_device)
                minkowski = minkowski.to(my_device)
                targets = targets.to(my_device)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = cnn(inputs, minkowski)
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
                        val_inputs, val_minkowski, val_targets = val_data
                        val_inputs = val_inputs.to(my_device)
                        val_minkowski = val_minkowski.to(my_device)
                        val_targets = val_targets.to(my_device)

                        val_outputs = cnn(val_inputs, val_minkowski)
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
            test_minkowski = dataset_test.minkowski.to(my_device)
            test_predictions = cnn(test_inputs, test_minkowski).cpu().numpy()

        test_predictions = scaler_permeability.inverse_transform(test_predictions.reshape(-1, 1)).reshape(-1)
        test_targets = scaler_permeability.inverse_transform(np.array(dataset_test.y).reshape(-1, 1)).reshape(-1)

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
