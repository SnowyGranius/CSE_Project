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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet()
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
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'splitter': ['best', 'random'],
            'max_depth': [5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'auto', 'sqrt', 'log2']
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
        'Ridge': {
            'alpha': [0.1, 1, 10]
        },
        'Lasso': {
            'alpha': [0.1, 1, 10]
        },
        'ElasticNet': {
            'alpha': [0.1, 1, 10],
            'l1_ratio': [0.1, 0.5, 0.9]
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



#pathlist = ['Eddie/Porespy_homogenous_diameter', 'Eddie/Heterogenous_samples', 'Eddie/Threshold_homogenous_diameter_small_RCP', 'Eddie/Threshold_homogenous_diameter_wide_RCP']
pathlist = ['Eddie/Porespy_homogenous_diameter']

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


    Bianca = automatic_process_regression(X_train, y_train, X_test, y_test)
    print(Bianca)