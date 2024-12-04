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

#for plotting
colorlist =['b', 'm', 'g', 'y', 'k']

def parity_plot(targets_original, targets_predicted, color, scaled, shape_dependence):
    if scaled == True:
        targets_predicted = scaler_y.inverse_transform(targets_predicted.reshape(-1, 1)).flatten()
        targets_original = scaler_y.inverse_transform(targets_original.reshape(-1, 1)).flatten()

    r2 = r2_score(targets_original, targets_predicted)
    mse = mean_squared_error(targets_original, targets_predicted)

    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())

    plt.scatter(targets_original, targets_predicted, alpha=0.6, color=color, label=f'R²: {r2:.5f}\nMSE (scaled data): {mse:.5f}')
    plt.plot([min(targets_original), max(targets_original)],
             [min(targets_original), max(targets_original)], color='r', linestyle='--', label='Parity Line')

    if shape_dependence == False:
        plt.title('Parity Plot '+ os.path.basename(path)+ ' Predictions Based on $M_0$, $M_1$ and $M_2$ ')
    else:
        plt.title('Parity Plot '+ os.path.basename(path)+ ' Predictions Based on $M_0$, $M_1$ and $M_2$ testing on '+ shape)

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

    parity_plot(y_actuals, y_predict, 'b', shape_dependence=True)
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

    # Calculate performance metrics
    mse = mean_squared_error(test_targets, predicted_labels)
    r2 = r2_score(test_targets, predicted_labels)

    print(f"Mean Squared Error: {mse:.10f}")
    print(f"R2 Score: {r2:.6f}")
    parity_plot(test_targets, predicted_labels, 'b', scaled=True, shape_dependence=True)
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

    # for i, term in enumerate(gam.terms):
    #     columns = ['Porosity', 'Surface', 'Euler_mean_vol', 'Permeability']
    #     if term.isintercept:
    #         continue
    #     XX = gam.generate_X_grid(term=i)
    #     plt.figure()
    #     plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX), label = columns[i])
    #     plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=0.95)[1], c='r', ls='--')
    #     plt.title(f'Feature {columns[i]}')
    #     plt.xlabel(columns[i])
    #     plt.ylabel('Partial Dependence')
    #     plt.show()

    # # Customize the plot
    # plt.title('Partial Dependence Plots')
    # plt.xlabel('Feature Value')
    # plt.ylabel('Partial Dependence')
    # plt.legend()  # Show legend for each feature
    # plt.grid(True)
    # plt.tight_layout()  # Optional: To make sure the layout doesn't overlap
    # plt.show()

    parity_plot(y_test, y_pred, 'b', scaled = False, shape_dependence=True)
    
    # plot_shapes(y_test, y_pred, 'b', marker_test)

    plt.show()

def decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeRegressor(max_depth=5) #max_depth=5
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_pred, y_test)
    mse = mean_squared_error(y_pred, y_test)
    
    # n_nodes = model.tree_.node_count
    # children_left = model.tree_.children_left
    # children_right = model.tree_.children_right
    # feature = model.tree_.feature
    # threshold = model.tree_.threshold
    # values = model.tree_.value
    
    # node_indicator = model.decision_path(X_test)
    # leaf_id = model.apply(X_test)

    # sample_id = 80
    
    # node_index = node_indicator.indices[
    # node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]]
    
    # print("Rules used to predict sample {id}:\n".format(id=sample_id))
    # for node_id in node_index:
    #     # continue to the next node if it is a leaf node
    #     if leaf_id[sample_id] == node_id:
    #         continue

    #     # check if value of the split feature for sample 0 is below threshold
    #     if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
    #         threshold_sign = "<="
    #     else:
    #         threshold_sign = ">"

    #     print(
    #         "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
    #         "{inequality} {threshold})".format(
    #             node=node_id,
    #             sample=sample_id,
    #             feature=feature[node_id],
    #             value=X_test[sample_id, feature[node_id]],
    #             inequality=threshold_sign,
    #             threshold=threshold[node_id],
    #         )
    #     )

    parity_plot(y_test, y_pred, color='b', scaled=True, shape_dependence=True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plot_tree(model, filled=True, feature_names=['Porosity', 'Surface', 'Euler_mean_vol'], 
              class_names=["Permeability"], rounded=True, fontsize=12,impurity=False, proportion=False)
    plt.title("Decision Tree Visualization")
    plt.show()


def SR_sindy_prediciton(X_train, y_train, X_test, y_test):
    feature_library = PolynomialLibrary(degree=3, include_interaction=True)

    model = SINDy(feature_library=feature_library)

    model.fit(X_train, t=y_train) 

    model.print()

def symbolic_regressor_model(X_train, y_train, X_test, y_test):
    converter = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    'neg': lambda x    : -x,
    'pow': lambda x, y : x**y,
    'inv': lambda x: 1/x,
    'sqrt': lambda x: x**0.5,
    'pow3': lambda x: x**3
    }

    def pow_3(x1):
        f = x1**3
        return f

    pow_3 = make_function(function=pow_3,name='pow3',arity=1)

    function_set = ['add', 'sub', 'mul', 'div','neg','inv',pow_3]

    start_time = time.time()
    symbolic_regressor = SymbolicRegressor(population_size=500, function_set=function_set,
                           generations=20, stopping_criteria=1e-8,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9,  metric='mse', verbose=1,
                           parsimony_coefficient=0.001, random_state=1234)
    
    symbolic_regressor.fit(X_train, y_train)
    y_pred = symbolic_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_pred, y_test)

    end_time = time.time()

    print(f"Time duration = {end_time - start_time}")

    print("Best program:", symbolic_regressor._program)
    print(f"Test MSE: {mse}")
    print(f"R² Score: {r2:.6f}")

    parity_plot(y_test, y_pred, 'b', scaled=True)
    plt.show()


def from_list_to_csv(X, y):
    data_sorted = np.hstack((X, y))

    data = np.hstack((X, y))

    columns = X.columns.tolist() + y.columns.tolist()

    df = pd.DataFrame(data_sorted, columns=columns)

    df.to_csv('output.csv', index=False)

def split_based_on_permeability(X, y, split_variable):
    sorted_indices = np.argsort(y.flatten())

    # Sort X and y using the sorted indices
    X = X[sorted_indices]
    y = y[sorted_indices]

    split_index = int(split_variable * len(X)) 

    # Split X and y into training and testing sets
    X_train = X[split_index:]
    y_train = y[split_index:]

    X_test = X[:split_index]
    y_test = y[:split_index]

    return X_train, X_test, y_train, y_test

def plot_shapes(targets_original, targets_predicted, color, marker_test):
    targets_predicted_scale = scaler_y.inverse_transform(targets_predicted.reshape(-1, 1)).flatten()
    targets_original_scale = scaler_y.inverse_transform(targets_original.reshape(-1, 1)).flatten()

    r2 = r2_score(targets_original, targets_predicted)
    mse = mean_squared_error(targets_original, targets_predicted)

    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())

    # plt.scatter(targets_original_scale, targets_predicted_scale, alpha=0.6, color=color, label=f'R²: {r2:.5f}\nMSE (scaled data): {mse:.5f}') 
    
    for i, marker in enumerate(marker_test):
        plt.scatter(
            targets_original_scale[i], 
            targets_predicted_scale[i], 
            alpha=0.6, 
            marker=marker,
            color='b'
        )
        
    plt.plot([min(targets_original_scale), max(targets_original_scale)],
             [min(targets_original_scale), max(targets_original_scale)], color='r', linestyle='--', label='Parity Line')

    plt.title('Parity Plot '+ os.path.basename(path)+ ' Predictions Based on $M_0$, $M_1$ and $M_2$ ')
    plt.xlabel('True Permeability')
    plt.ylabel('Predicted Permeability')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')


shapes = ['ellipse', 'circle', 'triangle', 'rectangle']

X_list = {
    'ellipse': [], 
    'circle': [], 
    'triangle': [], 
    'rectangle': []
}

X_scaled = {
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
                    
    # df_from_each_file = (pd.read_csv(f, dtype=np.float64) for f in all_files)
    # concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

    concatenated_df = pd.concat(data_list, ignore_index=True)

    X, y = concatenated_df[['Porosity', 'Surface', 'Euler_mean_vol']].values, concatenated_df['Permeability'].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    for shape in X_list:
        X_scaled_shape = scaler_X.fit_transform(X_list[shape])
        y_scaled_shape = scaler_y.fit_transform(y_list[shape])
        
    for shape in X_list:
        X_test = X_list[shape]
        y_test = y_list[shape]
        
        X_train = []
        y_train = []
        
        for shape2 in X_list:
            if shape2 != shape:
                X_train.extend(X_list[shape2])
                y_train.extend(y_list[shape2])
        
        decision_tree(X_train, y_train, X_test, y_test)



    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, marker_train, marker_test = train_test_split(X_scaled, y_scaled, marker_list, test_size=0.2, random_state=1234)

    # X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=1234)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    #X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = split_based_on_permeability(X_scaled, y_scaled, 0.2)

    #train_and_evaluate_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, n_epochs)

    #cross_validate_mlp(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, mlp, 5, True, n_epochs, batch_size, learning_rate)
    
    # GAM_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

    #decision_tree(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

    #SR_sindy_prediciton(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

    #symbolic_regressor_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

    # ellipse_exists = False

   


    

