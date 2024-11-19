import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import accuracy_score
import os
import sys

# NLLLoss function expects float64 dtype
torch.set_default_dtype(torch.float64)
# Assuming default device as 'cpu'. If GPU is desired, use 'cuda'
my_device = torch.device('cuda')

current_directory = os.path.dirname(os.path.abspath(sys.argv[0])) 
print(current_directory)

with np.load(f'{current_directory}/Datasets/mnist.npz') as data: # CHANGE TO OWN PATH TO DATA
    x_train = data['x_train']
    y_train = data['y_train'] 
    x_test= data['x_test'] 
    y_test = data['y_test']

print(f'Shape of training inputs: {x_train.shape}')
print(f'Shape of training labels: {y_train.shape}')
print(f'Shape of test inputs: {x_test.shape}')
print(f'Shape of test labels: {y_test.shape}')

# Helper function to visualize the MNIST image and its label on a subplot axis
def visualize_data(x,y,ax):
    ax.imshow(x, cmap='gray')
    ax.set(title = f'Label: {y}', xticks = [], yticks = [])

fig, axes = plt.subplots(2,4, layout='tight')
axs = axes.flatten()
random_ids = random.sample(list(range(x_test.shape[0])), 8)
for i, id in enumerate(random_ids):
    axs[i] = visualize_data(x_test[id,...], y_test[id], axs[i])

print(f'Minimum value in training images: {np.min(x_train)}')
print(f'Maximum value in training images: {np.max(x_train)}')

x_train = x_train / 255. 
x_test = x_test / 255.
print(f'Minimum value in training images: {np.min(x_train)}')
print(f'Maximum value in training images: {np.max(x_train)}')

x_mean = np.mean(x_train)
x_std = np.std(x_train)
print(f'Mean of x_train = {x_mean}') 
print(f'Std  of x_train = {x_std}')

class MNISTDataset(torch.utils.data.Dataset):
    '''
    MNIST dataset class for classification
    '''
    def __init__(self, X, y, scale_data = True):
        # Precomputed training set statistics for scaling
        self.X_mean = 0.1306604762738429
        self.X_std  = 0.3081078038564622
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            if scale_data:
                # Standardization: Z = (X - mean) / std
                X = (X - self.X_mean) / self.X_std
            # X is of shape [N, H, W], the CNN expects input [N, Channels, H, W]
            # The image is greyscale (contains only 1 channel), so we need to add a dummy channel dimension
            X = np.expand_dims(X, axis=1)
            # Convert to Pytorch tensor objects
            self.X = torch.tensor(X)
            self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
    # This is the method that returns the input, target data during model training
        return self.X[i], self.y[i]

class MNISTClassifierCNN(nn.Module):
    """
    CNN model for classification task on MNIST dataset
    """
    def __init__(self):
        # Use super to inherit nn.Module's methods
        super().__init__()
        # nn.Sequential is the basis for neural networks, indicates the sequential order of each NN layer
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
            # Apply 2x2 pooling operations to reduce image height and width by half
            nn.MaxPool2d(kernel_size=2),
            # Apply another convolution with more channels
            nn.Conv2d(10, 20, kernel_size=5),
            # Apply 2x2 pooling operations to reduce image height and width by half
            nn.MaxPool2d(kernel_size=2),
            # Now we flatten the output image [N, c, h, w] of the previous layer to [N, c*h*w] using a Flatten layer
            nn.Flatten(),
            # You have to precompute c*h*w to correctly define the MLP layers
            ############## MLP LAYERS #############
            # An input of 320 nodes to a layer of 64 nodes, 8 nodes is chosen for the 8 features of the dataset
            nn.Linear(320, 64),
            # ReLU activation function given for this layer
            nn.ReLU(),
            # Next layer takes in the 64 nodes, compresses to 32 nodes
            nn.Linear(64, 32),
            # ReLU activation function given for this layer
            nn.ReLU(),
            # Final layer takes 32 nodes and compresses to 10 nodes
            nn.Linear(32, 10),
            # ONLY FOR CLASSIFICATION TASK: we apply Log softmax activation. For regression tasks, you should use the output layer without activations
            nn.LogSoftmax(dim=1)
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
dataset_train = MNISTDataset(X=x_train, y=y_train, scale_data=True)
dataset_test = MNISTDataset(X=x_test, y=y_test, scale_data=True)

# Initialize the dataloader using batch size hyperparameter
batch_size = 32
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# Initialize the CNN
cnn = MNISTClassifierCNN()
print(cnn)
print(f'Number of learnable parameters in {cnn._get_name()} model = {count_parameters(cnn)}')
# Transfer model to your chosen device
cnn.to(my_device)

# Define the loss function and optimizer
## Neg-log-likelihood loss for classification task
loss_function = nn.NLLLoss()
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
        
        # Compute accuracy metric by converting the tensors to numpy
        ## Find the highest softmax probability to get the predicted label
        pred_labels = torch.argmax(outputs, dim=1).cpu().numpy()
        accuracy = accuracy_score(pred_labels, targets.detach().cpu().numpy())

        # Perform backwards pass
        loss.backward()

        # Perform optimization
        optimizer.step()
        
        # Log loss value per batch
        loss_per_batch.append(loss.item())
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