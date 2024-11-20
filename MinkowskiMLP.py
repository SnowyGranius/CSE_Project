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
import matplotlib.pyplot as plt


path = 'Heterogenous_samples'
all_files = glob.glob(os.path.join(path, '*.csv'))
df_from_each_file = (pd.read_csv(f, dtype=np.float64) for f in all_files)
concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

X, y = concatenated_df[['Porosity', 'Euler_mean_vol', 'Surface']].values, concatenated_df['Permeability'].values.reshape(-1, 1)

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
            nn.Linear(3, 64),
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


### Parity Plot


"""
import math

# Font for figure for publishing
font_axis_publish = {
        'color':  'black',
        'weight': 'bold',
        'size': 22,
        }
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16

# Read in data
#pred_vals = pd.read_csv("pred.csv", header=0, names=['Index','Pred'])
#gt_vals = pd.read_csv("gt.csv",header=0, names=['Index','GT'])

pred_vals = scaler_y.inverse_transform(predicted_labels.reshape(-1, 1)).flatten()
gt_vals = scaler_y.inverse_transform(test_targets.reshape(-1, 1)).flatten()

# Plot Figures
fignow = plt.figure(figsize=(8,8))

## find the boundaries of X and Y values
bounds = (min(gt_vals.min(), pred_vals.min()) - int(0.1 * pred_vals.min()), max(gt_vals.max(), pred_vals.max())+ int(0.1 * pred_vals.max()))

# Reset the limits
ax = plt.gca()
ax.set_xlim(bounds)
ax.set_ylim(bounds)
# Ensure the aspect ratio is square
ax.set_aspect("equal", adjustable="box")

plt.plot(gt_vals,pred_vals,"o", alpha=0.5 ,ms=10, markeredgewidth=0.0)

ax.plot([0, 1], [0, 1], "r-",lw=2 ,transform=ax.transAxes)

# Calculate Statistics of the Parity Plot 
mean_abs_err = np.mean(np.abs(gt_vals-pred_vals))
rmse = np.sqrt(np.mean((gt_vals-pred_vals)**2))
rmse_std = rmse / np.std(y)
z = np.polyfit(gt_vals,pred_vals, 1)
y_hat = np.poly1d(z)(gt_vals)

#text = f"$\: \: Mean \: Absolute \: Error \: (MAE) = {mean_abs_err:0.3f}$ \n $ Root \: Mean \: Square \: Error \: (RMSE) = {rmse:0.3f}$ \n $ RMSE \: / \: Std(y) = {rmse_std :0.3f}$ \n $R^2 = {r2_score(pred_vals,y_hat):0.3f}$"
text = f" "

plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')

# Title and labels 
plt.title("Parity Plot", fontdict=font_axis_publish)
plt.xlabel('Ground Truth', fontdict=font_axis_publish)
plt.ylabel('Prediction', fontdict=font_axis_publish)

# Save the figure into 300 dpi
#fignow.savefig("parityplot.png",format = "png",dpi=300,bbox_inches='tight')

#plt.show()
"""



# Transform data back to original scale
predicted_labels_original = scaler_y.inverse_transform(predicted_labels.reshape(-1, 1)).flatten()
test_targets_original = scaler_y.inverse_transform(test_targets.reshape(-1, 1)).flatten()


plt.figure(figsize=(8, 8))
plt.scatter(test_targets_original, predicted_labels_original, alpha=0.6, color='blue')
plt.plot([min(test_targets_original), max(test_targets_original)],
         [min(test_targets_original), max(test_targets_original)], color='red', linestyle='--', label='Parity Line')
plt.title('Parity Plot')
plt.xlabel('True Permeability')
plt.ylabel('Predicted Permeability')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()


