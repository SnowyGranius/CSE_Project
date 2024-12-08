import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import gpflow
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
import torch

print(tf.__version__)


np.random.seed(42)

def f_truth(x):
    # Return a sine
    #return torch.sin(2 * x)
    ## Use the following slightly more complex function when comparing between the predictions made with different kernels
    return 2*torch.sin(2*x) + torch.exp(x/5)


# The data is generated from the ground truth with i.i.d. gaussian noise
def f_data(x, rng, epsilon=0.2):
    # Generate N noisy observations (1 at each location)
    t = f_truth(x) + torch.normal(mean=0, std=epsilon, size=x.shape, generator=rng)

    return t

pathlist = ['Datasets/Porespy_homogeneous_diameter']
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
    K = C(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e2)) 
    
    #Matern kernel
    K_matern = C(1.0) * Matern(length_scale=10.0, nu=1.5) 
    
    gp = GaussianProcessRegressor(kernel=K_matern, alpha=1e-10, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(X_train, y_train.ravel())  
    
    y_pred, sigma = gp.predict(X_test, return_std=True)
    sigma_inv = sigma * scaler_y.scale_  # Scale the uncertainty back

    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Evaluate model
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    print(f"Mean Squared Error: {mse:.5f}")
    print(f"R² Score: {r2:.5f}")

    num_draws = 100  # Number of samples to draw from the posterior distribution
    posterior_samples = np.random.multivariate_normal(y_pred_inv, np.diag(sigma_inv ** 2), size=num_draws)
    plt.figure(figsize=(12, 6))
    for i in range(num_draws):
        plt.plot(X_test, posterior_samples[i], color='gray', alpha=0.2, label='Sample' if i == 0 else "")
    
    # Plot the mean prediction with the confidence interval
    plt.plot(X_test, y_pred_inv, color='blue', linewidth=3, label="Mean prediction")
    plt.fill_between(X_test.flatten(),
                     y_pred_inv - 1.96 * sigma_inv,
                     y_pred_inv + 1.96 * sigma_inv,
                     color='blue', alpha=0.3, label="95% Confidence Interval")

    plt.xlabel('Scaled Porosity (Input Feature)')
    plt.ylabel('Predicted Permeability')
    plt.title('Posterior Distribution of Gaussian Process Regression Predictions')
    plt.legend()
    plt.grid()
    plt.show()



