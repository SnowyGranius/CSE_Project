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
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

# Sci-kit learn imports
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, ARDRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor




def automatic_process_regression(X_train, y_train, X_test, y_test):
    
    models = {
        'LinearRegression': LinearRegression(), 
        'RandomForestRegressor': RandomForestRegressor(),
        'SVR': SVR(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'ExtraTreesRegressor': ExtraTreesRegressor(),
        'ARDRegressor': ARDRegression()
    }
    
    params = {
        'LinearRegression': {},
        'RandomForestRegressor': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
        },
        'SVR': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        },
        'DecisionTreeRegressor': {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
            'splitter': ['best', 'random'],
            'max_depth': [5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        },
        'KNeighborsRegressor': {
            'n_neighbors': [3, 5, 7, 9]
        },
        'GradientBoostingRegressor': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'AdaBoostRegressor': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'ExtraTreesRegressor': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        },
        'ARDRegressor': {
            'max_iter': [100, 200, 300]
        }
    }
    
    best_models = {}
    
    for model_name in models:
        grid_search = GridSearchCV(models[model_name], params[model_name], cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train.ravel())
        best_models[model_name] = grid_search.best_estimator_
        
    for model_name, model in best_models.items():
        y_pred = model.predict(X_test)
        #y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        mse = mean_squared_error(y_test, y_pred)
        print(f'{model_name} MSE: {mse}')
        
    return best_models


def ARDRegression_model(X_train, y_train, X_test, y_test):
    model = ARDRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'ARDRegression MSE: {mse}')
    print(f'Coefficients: {model.coef_}')
    
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(model.coef_)), model.coef_.flatten())
    plt.xlabel('Feature Index')
    plt.ylabel('Weight')
    plt.title('ARDRegression Model Weights')
    plt.show()
    """
    
    # Plot the true weights, the estimated weights, the histogram of the
    # weights, and predictions with standard deviations
    plt.figure(figsize=(6, 5))
    plt.title("Weights of the model")
    plt.plot(model.coef_, color="darkblue", linestyle="-", linewidth=2, label="ARD estimate")
    
    plt.xlabel("Features")
    plt.ylabel("Values of the weights")
    plt.legend(loc=1)

    """
    n_features = X_train.shape[1]

    plt.figure(figsize=(6, 5))
    plt.title("Histogram of the weights")
    plt.hist(model.coef_, bins=n_features, color="navy", log=True)
    plt.scatter(
        model.coef_[relevant_features],
        np.full(len(relevant_features), 5.0),
        color="gold",
        marker="o",
        label="Relevant features",
    )
    plt.ylabel("Features")
    plt.xlabel("Values of the weights")
    plt.legend(loc=1)
    """

    plt.figure(figsize=(6, 5))
    plt.title("Marginal log-likelihood")
    plt.plot(model.scores_, color="navy", linewidth=2)
    plt.ylabel("Score")
    plt.xlabel("Iterations")
    
    plt.show()
    
    return model


def ARDRegression_model_optimized(X_train, y_train, X_test, y_test):
    model = ARDRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'ARDRegression MSE: {mse}')
    print(f'Coefficients: {model.coef_}')
    
    # Plot the weights
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(model.coef_.flatten())), model.coef_.flatten())
    plt.xlabel('Feature Index')
    plt.ylabel('Weight')
    plt.title('ARDRegression Model Weights')
    plt.show()
    
    # Plot the true weights, the estimated weights, the histogram of the weights, and predictions with standard deviations
    plt.figure(figsize=(6, 5))
    plt.title("Weights of the model")
    plt.plot(model.coef_, color="darkblue", linestyle="-", linewidth=2, label="ARD estimate")
    plt.xlabel("Features")
    plt.ylabel("Values of the weights")
    plt.legend(loc=1)
    plt.show()

    n_features = X_train.shape[1]  # Number of features in the dataset

    """
    plt.figure(figsize=(6, 5))
    plt.title("Histogram of the weights")
    plt.hist(model.coef_, bins=n_features, color="navy", log=True)
    plt.ylabel("Frequency")
    plt.xlabel("Values of the weights")
    plt.show()
    """

    """
    # Plot predictions with standard deviations
    plt.figure(figsize=(6, 5))
    plt.title("Predictions with standard deviations")
    yerr = model.sigma_.flatten() if model.sigma_.ndim > 1 else model.sigma_
    yerr = yerr[:len(y_test.flatten())]  # Ensure yerr matches the length of y_test
    plt.errorbar(range(len(y_test.flatten())), y_test.flatten(), yerr=yerr, fmt='o', label='True values')
    plt.errorbar(range(len(y_pred.flatten())), y_pred.flatten(), yerr=yerr, fmt='x', label='Predicted values')
    plt.xlabel("Sample index")
    plt.ylabel("Permeability")
    plt.legend(loc=1)
    plt.show()
    """
    
    return model



pathlist = ['Eddie/Porespy_homogenous_diameter', 'Eddie/Heterogenous_samples', 'Eddie/Threshold_homogenous_diameter_small_RCP', 'Eddie/Threshold_homogenous_diameter_wide_RCP']
#pathlist = ['Eddie/Porespy_homogenous_diameter']

use_all = False

if use_all is False:
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


        best_models = automatic_process_regression(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
        print(best_models)
        
        #ARD_results = ARDRegression_model_optimized(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
        #print(ARD_results)
    
else:
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
    
    best_models = automatic_process_regression(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
    print(best_models)
        
    ARD_results = ARDRegression_model_optimized(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
    print(ARD_results)