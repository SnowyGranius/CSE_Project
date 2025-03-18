import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os
import glob
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM, s
from sklearn.tree import DecisionTreeRegressor, plot_tree
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from gplearn.genetic import SymbolicRegressor
from sympy import *
from gplearn.functions import make_function


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

def train_and_evaluate_model(X_train, y_train, X_test, y_test, n_epochs):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for batch processing
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)

    for epoch in n_epochs:  # 5 epochs
        #print(f'Starting epoch {epoch + 1}')
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
    
    return test_targets, predicted_labels

#for plotting
colorlist =['b', 'm', 'g', 'y', 'k']

font_size = 14
fig_size = (9, 7)


shapes = ['ellipse', 'circle', 'triangle', 'rectangle']

X_list = {
    'ellipse': [], 
    'circle': [], 
    'triangle': [], 
    'rectangle': []
}

X_scaled_shape = {
    'ellipse': [], 
    'circle': [], 
    'triangle': [], 
    'rectangle': []
}

y_scaled_shape = {
    'ellipse': [], 
    'circle': [], 
    'triangle': [], 
    'rectangle': []
}

y_list = {
    'ellipse': [], 
    'circle': [], 
    'triangle': [], 
    'rectangle': []
}

markers = ['d', 'o', '^', 's']

#might have to change this path in Git
pathlist = ['Datasets/Porespy_homogeneous_diameter']
#, 'Datasets/Heterogeneous_samples', 'Datasets/Threshold_homogenous_diamater_small_RCP', 'Datasets/Threshold_homogenous_diamater_wide_RCP']


for path in pathlist:
    all_files = glob.glob(os.path.join(path, '*.csv'))
    
    data_list = []
    shape_list = []
    marker_list = []
    
    # for f in all_files:
    for i in range(len(shapes)):
        data_list_shape =[]
        for f in all_files:
            if shapes[i] in f:
                df = pd.read_csv(f, dtype=np.float64)
                data_list.append(df)
                data_list_shape.append(df)
                shape_list.extend([shapes[i]] * len(df))
                marker_list.extend([markers[i]] * len(df))
                
        concatenated_df = pd.concat(data_list_shape, ignore_index=True)
        
        X_list[shapes[i]]= (concatenated_df[['Porosity', 'Surface', 'Euler_mean_vol']].values)
        y_list[shapes[i]]= (concatenated_df['Permeability'].values.reshape(-1,1))
                    
    df_from_each_file = (pd.read_csv(f, dtype=np.float64) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

    # concatenated_df = pd.concat(data_list, ignore_index=True)
    X, y = concatenated_df[['Porosity', 'Surface', 'Euler_mean_vol']].values, concatenated_df['Permeability'].values.reshape(-1, 1)
    print(np.size(y))
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    for shape in shapes:
        X_scaled_shape[shape] = scaler_X.fit_transform(X_list[shape])
        y_scaled_shape[shape] = scaler_y.fit_transform(y_list[shape])
        
    # fig, axes = plt.subplots(2, 2, figsize=(9,7.5), sharey=False)  # Adjust to 2x2 for 4 shapes
    # axes = axes.ravel()  # Flatten for easier indexing

    for i, shape in enumerate(shapes):
        print(f"Prediction using {shape}")
        
        X_train = X_scaled_shape[shape]
        y_train = y_scaled_shape[shape]
        
        print(f"Train elem: {np.shape(X_train)}")
        
        X_test = []
        y_test = []
        
        for shape2 in shapes:
            if shape2 != shape:
                X_test.extend(X_scaled_shape[shape2])
                y_test.extend(y_scaled_shape[shape2])
        
        print(f"Test elem: {np.shape(X_test)}")
        
        # Train and get predictions
        targets_original, targets_predicted = train_and_evaluate_model(X_train, y_train, X_test, y_test, n_epochs)
        
        # Scale predictions back to the original scale
        targets_predicted = scaler_y.inverse_transform(targets_predicted.reshape(-1, 1)).flatten()
        targets_original = scaler_y.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
        
        # Compute metrics
        r2 = r2_score(targets_original, targets_predicted)
        mse = mean_squared_error(targets_original, targets_predicted)
        
        print(f"Mean Squared Error: {mse:.15f}")
        print(f"R2 Score: {r2:.6f}")
        
        plt.figure(figsize=(8,8))
        
        # Parity plot in the corresponding subplot
        # ax = axes[i]
        plt.scatter(targets_original, targets_predicted, alpha=0.6, color='blue',
                label=f'R²: {r2:.5f}\nMSE: {mse:.5e}')
        plt.plot([min(targets_original), max(targets_original)],
                [min(targets_original), max(targets_original)], color='red', linestyle='--', label='Parity Line')
        
        # Configure subplot
        plt.title(f"Parity Plot MLP Training on {shape.capitalize()}", fontsize=16)
        plt.xlabel("True Permeability", fontsize=14)
        plt.ylabel("Predicted Permeability", fontsize=14)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True)
        plt.axis('equal')
        # plt.tight_layout()
        plt.savefig(f"Parity Plot MLP Training on {shape.capitalize()}")

        plt.show()
        
        # Format axes for scientific notation
        # ax.yaxis.set_major_formatter(ScalarFormatter())
        # ax.xaxis.set_major_formatter(ScalarFormatter())

    # Adjust layout and show the plot
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f"Parity Plot MLP Training on {shape.capitalize()}")