import torch
import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
import time

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sympy import *
from pygam import LinearGAM, s
from gplearn.functions import make_function

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

# Sci-kit learn imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor




#arhitecture of MLP
class MLP(nn.Module):
    '''
     Multilayer perceptron for regression
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 32),
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

    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())

    plt.scatter(targets_original_scale, targets_predicted_scale, alpha=0.6, color=color, label=f'R²: {r2:.5f}\nMSE (scaled data): {mse:.5f}')
    plt.plot([min(targets_original_scale), max(targets_original_scale)],
             [min(targets_original_scale), max(targets_original_scale)], color='r', linestyle='--', label='Parity Line')

    plt.title('Parity Plot '+ os.path.basename(path)+ ' Predictions Based on $M_0$, $M_1$ and $M_2$ ')
    plt.xlabel('True Permeability')
    plt.ylabel('Predicted Permeability')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

def cross_validate_mlp(X_train_init, y_train_init, X_test_init, y_test_init, model, k, shuffle, n_epochs, batch_size, learning_rate):
    start_time = time.time()

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

    end_time = time.time()

    print(f"time duration = {end_time - start_time}")

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


def GAM_model(X_train, y_train, X_test, y_test): 
    gam = LinearGAM(s(0, lam=0.6) + s(1, lam=0.6) + s(2, lam=0.6))  # s(i) defines a smooth term for feature i
    gam.fit(X_train, y_train) #gridsearch not that good

    y_pred = gam.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error Test (MSE): {mse:.15f}")
    print(f"R² Score Test: {r2:.6f}")

    y_pred_train = gam.predict(X_train)

    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    print(f"Mean Squared Error Train (MSE): {mse_train:.15f}")
    print(f"R² Score Train: {r2_train:.6f}")

    for i, term in enumerate(gam.terms):
        df = concatenated_df[['Porosity', 'Surface', 'Euler_mean_vol', 'Permeability']]
        if term.isintercept:
            continue
        XX = gam.generate_X_grid(term=i)
        plt.figure()
        plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
        plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=0.95)[1], c='r', ls='--')
        plt.title(f'Feature {df.columns[i]}')
        plt.xlabel(df.columns[i])
        plt.ylabel('Partial Dependence')
        plt.show()

    parity_plot(y_test, y_pred, 'b')

    plt.show()

def decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # parity_plot(y_test, y_pred, 'b')
    r2 = r2_score(y_pred, y_test)
    mse = mean_squared_error(y_pred, y_test)

    print(f"Mean Squared Error (MSE): {mse:.15f}")
    print(f"R² Score: {r2:.6f}")

    plt.scatter(y_test, y_pred, alpha=0.6, color='b', label=f'R²: {r2:.5f}\nMSE: {mse:.5f}')
    plt.plot([min(y_test), max(y_test)],
             [min(y_test), max(y_test)], color='r', linestyle='--', label='Parity Line')

    plt.title('Parity Plot '+ os.path.basename(path)+ ' Predictions Based on $M_0$, $M_1$ and $M_2$')
    plt.xlabel('True Permeability')
    plt.ylabel('Predicted Permeability')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.show()



#pathlist = ['Eddie/Porespy_homogenous_diameter', 'Eddie/Heterogenous_samples', 'Eddie/Threshold_homogenous_diameter_small_RCP', 'Eddie/Threshold_homogenous_diameter_wide_RCP']
pathlist = ['Eddie/CirclesMLP']

for path in pathlist:
    all_files = glob.glob(os.path.join(path, '*.csv'))
    df_from_each_file = (pd.read_csv(f, dtype=np.float64) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

    X, y = concatenated_df[['Porosity', 'Surface', 'Euler_mean_vol']].values, concatenated_df['Permeability'].values.reshape(-1, 1)
   
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=1234)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


    train_and_evaluate_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, n_epochs)

    #cross_validate_mlp(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, mlp, 5, True, n_epochs, batch_size, learning_rate)
    
    #GAM_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

    #decision_tree(X_train, y_train, X_test, y_test)


