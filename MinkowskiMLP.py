import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os
import glob
from sklearn.preprocessing import StandardScaler


path = 'Datasets/Porespy_homogenous_diamater'
all_files = glob.glob(os.path.join(path, '*.csv'))
df_from_each_file = (pd.read_csv(f, dtype=np.float64) for f in all_files)
concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

X, y = concatenated_df[['Porosity', 'Euler_mean_vol']].values, concatenated_df['Permeability'].values.reshape(-1, 1)

# Scale the features and targets
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=1234)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
class MLP(nn.Module):
    '''
    Multilayer perceptron for regression
    '''
    def __init__(self):
        # Use super to inherit nn.Module's methods
        super().__init__()
        # nn.Sequential is the basis for neural networks, indicates the sequential order of each
        # NN layer
        self.layers = nn.Sequential(
            # An input of 8 nodes to a layer of 64 nodes, 8 nodes is chosen for the 8 features of
            # the dataset
            nn.Linear(2, 64),
            # ReLU activation function given for this layer
            nn.ReLU(),
            # Next layer takes in the 64 nodes, compresses to 32 nodes
            nn.Linear(64, 32),
            # ReLU activation function given for this layer
            nn.ReLU(),
            # Final layer takes 32 nodes and compresses to one output, the regression target value
            nn.Linear(32, 1)
        )
    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)


mlp = MLP()

# Define the loss function and optimizer
learning_rate = 1e-2
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

for epoch in range(100):  # 5 epochs
    print(f'Starting epoch {epoch + 1}')
    current_loss = 0.0
    print("X_train shape:", X_train.shape)

    for inputs, targets in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = mlp(inputs)

        # Compute loss
        loss = loss_function(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        current_loss += loss.item()

    print(f'Epoch {epoch + 1} - Loss: {current_loss / len(train_loader):.6f}')

print('Training process has finished.')

# Evaluate model
mlp.eval()
with torch.no_grad():
    test_outputs = mlp(X_test)
    predicted_labels = test_outputs.squeeze().tolist()

# Convert predictions and actual values to NumPy arrays
predicted_labels = np.array(predicted_labels)
test_targets = y_test.squeeze().numpy()

# Calculate performance metrics
mse = mean_squared_error(test_targets, predicted_labels)
r2 = r2_score(test_targets, predicted_labels)
print(f"Mean Squared Error: {mse:.10f}")
print(f"R2 Score: {r2:.6f}")