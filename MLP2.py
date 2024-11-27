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
from sklearn.linear_model import LinearRegression

#arhitecture of MLP
class MLP(nn.Module):
    '''
     Multilayer perceptron for regression
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            # nn.Linear(32, 16),
            # nn.LeakyReLU(),
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

#for plotting
colorlist =['b', 'm', 'g', 'y', 'k']

def parity_plot(targets_original, targets_predicted, color):
    targets_predicted_scale = scaler_y.inverse_transform(targets_predicted.reshape(-1, 1)).flatten()
    targets_original_scale = scaler_y.inverse_transform(targets_original.reshape(-1, 1)).flatten()

    r2 = r2_score(targets_original, targets_predicted)
    mse = mean_squared_error(targets_original, targets_predicted)

    plt.scatter(targets_original_scale, targets_predicted_scale, alpha=0.6, color=color, label=f'R²: {r2:.5f}\nMSE: {mse:.5f}')
    plt.plot([min(targets_original_scale), max(targets_original_scale)],
             [min(targets_original_scale), max(targets_original_scale)], color='r', linestyle='--', label='Parity Line')
             
    plt.title('Parity Plot '+ os.path.basename(path)+ ' Predictions Based on $M_0$ and $M_1$')
    plt.xlabel('True Permeability')
    plt.ylabel('Predicted Permeability')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

def cross_validate_mlp(X_train_init, y_train_init, X_test_init, y_test_init, model, k, shuffle, n_epochs, batch_size, learning_rate):
    kf = KFold(n_splits=k, shuffle=shuffle)
    mse_scores, r2_scores = [], []
    mse_scores_train, r2_scores_train = [], []

    all_predictions = []
    all_actuals = []

    # Scale the features and targets
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

        # Evaluate model
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val).squeeze().numpy()
            val_predictions_train =  model(X_train).squeeze().numpy()

        val_actuals = np.array(y_val)
        val_actuals_train = np.array(y_train)

        #parity_plot(val_actuals, val_predictions, colorlist[fold])

        mse = mean_squared_error(val_actuals, val_predictions)
        r2 = r2_score(val_actuals, val_predictions)

        mse_train = mean_squared_error(val_actuals_train, val_predictions_train)
        r2_train = r2_score(val_actuals_train, val_predictions_train)

        mse_scores.append(mse)
        r2_scores.append(r2)

        mse_scores_train.append(mse_train)
        r2_scores_train.append(r2_train)

        all_predictions.extend(val_predictions)
        all_actuals.extend(val_actuals)

        print(f"Fold {fold + 1} - MSE test: {mse:.10f}, R²: {r2:.4f}")
        print(f"Fold {fold + 1} - MSE train: {mse_train:.10f}, R²: {r2_train:.4f}")

    X_train_init = torch.tensor(X_train_init, dtype=torch.float32)
    y_train_init = torch.tensor(y_train_init, dtype=torch.float32)
   
    with torch.no_grad():
        y_predict = model(X_test_init).squeeze().numpy()
        y_predict_train = model(X_train_init).squeeze().numpy()
            
    y_actuals = np.array(y_test_init)
    y_actuals_train = np.array(y_train_init)

    # predicted_labels_original = scaler_y.inverse_transform(y_predict.reshape(-1, 1)).flatten()
    # test_targets_original = scaler_y.inverse_transform(y_actuals.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_actuals, y_predict)
    r2 = r2_score(y_actuals, y_predict)

    mse_train = mean_squared_error(y_actuals_train, y_predict_train)
    r2_train = r2_score(y_actuals_train, y_predict_train)

    print(f"Final MSE Test: {mse:.6f}")
    print(f"Final R² Test: {r2:.4f}")

    print(f"Final MSE Train: {mse_train:.6f}")
    print(f"Final R² Train: {r2_train:.4f}")

    parity_plot(y_actuals, y_predict, 'b')
    plt.show()

def train_and_evaluate_model(X_train, y_train, X_test, y_test, n_epochs):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for batch processing
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)

    for epoch in n_epochs:  # 5 epochs
        print(f'Starting epoch {epoch + 1}')
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

    # Evaluate model
    mlp.eval()
    with torch.no_grad():
        predicted_labels = mlp(X_test).squeeze().numpy()

    test_targets = y_test.squeeze().numpy()

    # Calculate performance metrics
    mse = mean_squared_error(test_targets, predicted_labels)
    r2 = r2_score(test_targets, predicted_labels)
    print(f"Mean Squared Error: {mse:.10f}")
    print(f"R2 Score: {r2:.6f}")
    parity_plot(test_targets, predicted_labels, 'b')
    plt.show()

def linear_regression_prediciton(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"R² Score: {r2:.6f}")

    parity_plot(y_test, y_pred, 'b')
    plt.show()

pathlist = ['Datasets/Porespy_homogenous_diamater', 'Datasets/Heterogeneous_samples', 'Datasets/Threshold_homogenous_diamater_small_RCP', 'Datasets/Threshold_homogenous_diamater_wide_RCP']

for path in pathlist:
    all_files = glob.glob(os.path.join(path, '*.csv'))
    df_from_each_file = (pd.read_csv(f, dtype=np.float64) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

    X, y = concatenated_df[['Porosity', 'Surface']].values, concatenated_df['Permeability'].values.reshape(-1, 1)
   
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=1234)

    # cross_validate_mlp(X_train, y_train, X_test, y_test, mlp, 5, True, n_epochs, batch_size, learning_rate)
    
    linear_regression_prediciton(X_train, y_train, X_test, y_test)

    # ellipse_exists = False

    # for f in all_files:
    #     if 'ellipse' in f:
    #         ellipse_exists = True

    # if ellipse_exists == True: 
    #     files_ellipse = [f for f in all_files if 'ellipse' in f]
    #     df_from_ellipse = (pd.read_csv(f, dtype=np.float64) for f in files_ellipse)
    #     concatenated_df_ellipse = pd.concat(df_from_ellipse, ignore_index=True)
        
    #     X_ellipse, y_ellipse = concatenated_df_ellipse[['Porosity', 'Euler_mean_vol', 'Surface']].values, concatenated_df_ellipse['Permeability'].values.reshape(-1, 1)

    #     X_scaled_ellipse = scaler_X.fit_transform(X_ellipse)
    #     y_scaled_ellipse = scaler_y.fit_transform(y_ellipse)

    #     X_train_ellipse, X_test_ellipse, y_train_ellipse, y_test_ellipse = train_test_split(X_scaled_ellipse, y_scaled_ellipse, test_size=0.2, random_state=1234)

    #     train_and_evaluate_model(X_train_ellipse, y_train_ellipse, X_test_ellipse, y_test_ellipse, n_epochs)




    

