import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import glob
import torch
import time as time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

def plot_preds(mu_pred, X_train, X_test, y_train, posterior_samples, domain, confidence_upper, confidence_lower):
    """
    Plot GP predictions with confidence intervals and posterior samples.
    
    Parameters:
    - mu_pred: Mean predictions from the GP.
    - X_train: Training data (in original scale).
    - X_test: Test data (in original scale).
    - y_train: Training labels (in original scale).
    - posterior_samples: Samples from the posterior distribution.
    - domain: Range of x values for plotting.
    - confidence_upper: Upper bound of confidence intervals.
    - confidence_lower: Lower bound of confidence intervals.
    """

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    # Plot the distribution of the function (mean, covariance)
    ax1.plot(X_test, mu_pred, "C0-", label="GP Mean")
    ax1.fill_between(X_test.flatten(), confidence_lower, confidence_upper, color="C0", alpha=0.2, label="95% CI")
    ax1.scatter(X_train, y_train, color="red", label="Training Points")
    ax1.set_xlabel("$M_1$", fontsize=13)
    ax1.set_ylabel("Permeability", fontsize=13)
    ax1.set_title("GP Predictions with 95% Confidence Interval")
    ax1.legend()
    ax1.grid()

     # Posterior samples
    for sample in posterior_samples:
        ax2.plot(X_test, sample, alpha=0.6)
    ax2.set_xlabel("$M_1$", fontsize=13)
    ax2.set_ylabel("Permeability", fontsize=13)
    ax2.set_title("Posterior Samples")
    ax2.grid()
    plt.tight_layout()
    plt.savefig('GP_Confidence_Interval')
    plt.show()
    
# Evaluate different combinations of hyperparameters
best_mse = float('inf')
best_params = {}

pathlist = ['Porespy_homogenous_diameter']
#, 'Datasets/Heterogeneous_samples', 'Datasets/Threshold_homogenous_diamater_small_RCP', 'Datasets/Threshold_homogenous_diamater_wide_RCP']

    
for path in pathlist:
    all_files = glob.glob(os.path.join(path, '*.csv'))
    df_from_each_file = (pd.read_csv(f, dtype=np.float64) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    
    X, y = concatenated_df[['Surface']].values, concatenated_df['Permeability'].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=1234)
            
    #RBF kernel
    K = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X_train.shape[1]), length_scale_bounds=(0.001, 10)) 
    
    #Matern kernel
    K_matern = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X_train.shape[1]), length_scale_bounds=(0.001, 10), nu=1.5) 
    
    #Radial quadratic kernel
    K_rq = C(1.0) * RationalQuadratic(length_scale=10.0, alpha=1.5)
    
    start_time = time.time()
    
    gp = GaussianProcessRegressor(kernel=K, alpha=0.01, n_restarts_optimizer=20, normalize_y=True)
    gp.fit(X_train, y_train)  
    
    y_pred, sigma = gp.predict(X_test, return_std=True)
    
    end_time = time.time()
    
    print(f"Time duration: {end_time - start_time}")
    
    mu_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    sigma_scaled = sigma * scaler_y.scale_
    confidence_upper = mu_pred + 1.96 * sigma_scaled
    confidence_lower = mu_pred - 1.96 * sigma_scaled
    
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    X_train_inv = scaler_X.inverse_transform(X_train)
    X_test_inv = scaler_X.inverse_transform(X_test)
    y_train_inv = scaler_y.inverse_transform(y_train).flatten()


    # Evaluate model
    mse = mean_squared_error(y_test_inv, mu_pred)
    r2 = r2_score(y_test_inv, mu_pred)
    print(f"Mean Squared Error: {mse:.6e}")
    print(f"R² Score: {r2:.5f}")
    
    plt.figure(figsize=(10, 7))

    plt.scatter(y_test_inv, mu_pred, alpha=0.6, color='b', label=f'R²: {r2:.5f}\nMSE (scaled data): {mse:.3e}')
    plt.plot([min(y_test_inv), max(y_test_inv)],
             [min(y_test_inv), max(y_test_inv)], color='r', linestyle='--', label='Parity Line')

    plt.title('Parity Plot '+ os.path.basename(path)+ ' Gaussian Process Prediciton Based on $M_0$, $M_1$ and $M_2$')

    plt.xlabel('True Permeability')
    plt.ylabel('Predicted Permeability')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.savefig('Parity_GP', dpi=300)
    plt.show()
    
    X_test = np.linspace(np.min(X),np.max(X),2000).reshape(-1,1)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_pred, sigma = gp.predict(X_test_scaled, return_std=True)
    mu_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    sigma_scaled = sigma * scaler_y.scale_
    confidence_upper = mu_pred + 1.96 * sigma_scaled
    confidence_lower = mu_pred - 1.96 * sigma_scaled
    
    posterior_samples = gp.sample_y(X_test_scaled, n_samples=12).T
    posterior_samples_inv = scaler_y.inverse_transform(posterior_samples)
    
    domain = [X_test.min(), X_test.max()]

    # Plot predictions
    domain = [X_test_inv.min(), X_test_inv.max()]
    plot_preds(mu_pred, X_train_inv, X_test, y_train_inv, posterior_samples_inv, domain, confidence_upper, confidence_lower)