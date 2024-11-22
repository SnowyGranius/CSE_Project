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
import seaborn as sns
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import KFold

#arhitecture of MLP
class MLP(nn.Module):
    '''
     Multilayer perceptron for regression
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)

mlp = MLP()

n_epochs = np.arange(1, 200, 1) 

batch_size = 32

# Define the loss function and optimizer
learning_rate = 1e-2
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

colorlist =['b', 'm', 'g', 'y', 'k']

def parity_plot(targets_original, targets_predicted, color):
    plt.scatter(targets_original, targets_predicted, alpha=0.6, color=color)
    plt.plot([min(targets_original), max(targets_original)],
             [min(targets_original), max(targets_original)], color=color, linestyle='--', label='Parity Line')
    plt.title('Parity Plot '+ os.path.basename(path))
    plt.xlabel('True Permeability')
    plt.ylabel('Predicted Permeability')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

pathlist = ['Porespy_homogenous_diamater']

for path in pathlist:

    all_files = glob.glob(os.path.join(path, '*.csv'))
    df_from_each_file = (pd.read_csv(f, dtype=np.float64) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

    X, y = concatenated_df[['Porosity', 'Euler_mean_vol', 'Surface']].values, concatenated_df['Permeability'].values.reshape(-1, 1)

    k = 2

    kf = KFold(n_splits=k, shuffle=True)
    mse_scores, r2_scores = [], []

    all_predictions = []
    all_actuals = []

    # Scale the features and targets
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train_init, X_test_init, y_train_init, y_test_init = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=1234)

    X_test_init = torch.tensor(X_test_init, dtype=torch.float32)
    y_test_init = torch.tensor(y_test_init, dtype=torch.float32)

    plt.figure(figsize=(8, 8))

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_init)):
        print(f"Starting Fold {fold + 1}")

        # Split data
        X_train, X_val = X_train_init[train_index], X_train_init[val_index]
        y_train, y_val = y_train_init[train_index], y_train_init[val_index]

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in n_epochs:  
            # print(f'Starting epoch {epoch + 1}')
            current_loss = 0.0

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

                # mlp.eval()  # Set model to evaluation mode
                # with torch.no_grad():
                #     val_outputs = mlp(X_val)
                #     val_loss = loss_function(val_outputs, y_val)
            
                # print(f'Epoch {epoch + 1} - Training Loss: {current_loss / len(train_loader):.6f} - Validation Loss: {val_loss.item():.6f}')


                # print(f'Epoch {epoch + 1} - Loss: {current_loss / len(train_loader):.6f}')

            # print('Training process has finished.')

            # Evaluate model
        mlp.eval()
        with torch.no_grad():
            val_outputs = mlp(X_val).squeeze().numpy()
            val_predictions = val_outputs.squeeze().tolist()
           
        val_predictions = np.array(val_predictions)
        val_actuals = np.array(y_val)
        

        mse = mean_squared_error(val_actuals, val_predictions)
        r2 = r2_score(val_actuals, val_predictions)

        mse_scores.append(mse)
        r2_scores.append(r2)

        all_predictions.extend(val_predictions)
        all_actuals.extend(val_actuals)

        print(f"Fold {fold + 1} - MSE: {mse:.10f}, R²: {r2:.4f}")
        
    with torch.no_grad():
        y_final_predict = mlp(X_test_init).squeeze().numpy()
        y_final_predict = y_final_predict.squeeze().tolist()
           
    y_final_predict = np.array(y_final_predict)
    y_actuals = np.array(y_test_init)

    predicted_labels_original = scaler_y.inverse_transform(y_final_predict.reshape(-1, 1)).flatten()
    test_targets_original = scaler_y.inverse_transform(y_actuals.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_actuals, y_final_predict)
    r2 = r2_score(y_actuals, y_final_predict)

    print(f"Final MSE: {mse:.6f}")
    print(f"Final R²: {r2:.4f}")

    parity_plot(test_targets_original, predicted_labels_original, 'b')
    plt.show()
   


# sangele nu foloseste inca asta
def cross_validate_mlp(X, y, model_class, k, n_epochs, batch_size, learning_rate):
    kf = KFold(n_splits=k, shuffle=True, random_state=1234)
    mse_scores, r2_scores = [], []

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))


    for fold, (train_index, val_index) in enumerate(kf.split(X_scaled)):
        print(f"Starting Fold {fold + 1}")

        # Split data
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y_scaled[train_index], y_scaled[val_index]

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        # Create DataLoader for batch processing
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model, optimizer, and loss function
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = torch.nn.L1Loss()

        # Training loop
        for epoch in range(n_epochs):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze().numpy()
            val_targets = y_val.squeeze().numpy()
            val_predictions = scaler_y.inverse_transform(val_outputs.reshape(-1, 1)).squeeze()
            val_actuals = scaler_y.inverse_transform(val_targets.reshape(-1, 1)).squeeze()

        # Metrics
        mse = mean_squared_error(val_actuals, val_predictions)
        r2 = r2_score(val_actuals, val_predictions)

        mse_scores.append(mse)
        r2_scores.append(r2)

        all_predictions.extend(val_predictions)
        all_actuals.extend(val_actuals)

        print(f"Fold {fold + 1} - MSE: {mse:.6f}, R²: {r2:.4f}")



