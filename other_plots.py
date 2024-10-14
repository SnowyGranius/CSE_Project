import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.interpolate import make_interp_spline
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy.optimize as opt

# A program that reads data from a csv file and compares features using a pairplot

# # Read the data from the csv file
# df = pd.read_csv('C:\\Users\\ionst\\Documents\\Fisiere Python\\Proiect\\Data\Datasets\\Heterogenous_samples\\Minkowskis_pf_0.220_rectangle_1_hetero.csv')
# print(df)

# Read all relevant files in the folder

path = 'C:\\Users\\ionst\\Documents\\Fisiere_Python\\Proiect\\Data\\Datasets\\Porespy_homogenous_diamater\\'
# all_files = glob.glob(os.path.join(path, "*0.300*.csv"))
all_files = glob.glob(os.path.join(path, "*rectangle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_rectangle = pd.concat(df_from_each_file, ignore_index=True)
concatenated_df_rectangle['Euler_total'] = 'Rectangle'
# concatenated_df_rectangle['Permeability'] = concatenated_df_rectangle['Permeability'] * 1e5
# concatenated_df_rectangle['Energy'] = concatenated_df_rectangle['Energy'] * 1e5


all_files = glob.glob(os.path.join(path, "*triangle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_triangle = pd.concat(df_from_each_file, ignore_index=True)
concatenated_df_triangle['Euler_total'] = 'Triangle'
# concatenated_df_triangle['Permeability'] = concatenated_df_triangle['Permeability'] * 1e5
# concatenated_df_triangle['Energy'] = concatenated_df_triangle['Energy'] * 1e5


all_files = glob.glob(os.path.join(path, "*ellipse*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_ellipse = pd.concat(df_from_each_file, ignore_index=True)
concatenated_df_ellipse['Euler_total'] = 'Ellipse'
# concatenated_df_ellipse['Permeability'] = concatenated_df_ellipse['Permeability'] * 1e5
# concatenated_df_ellipse['Energy'] = concatenated_df_ellipse['Energy'] * 1e5



# concated_df = pd.concat([concatenated_df_rectangle, concatenated_df_triangle, concatenated_df_ellipse], ignore_index=True)
# concated_df = pd.concat([concatenated_df_rectangle, concatenated_df_triangle], ignore_index=True)
concated_df = pd.concat([concatenated_df_rectangle], ignore_index=True)
#print(concated_df)


def kozeny_carman(lst, gamma):
    return gamma*lst[0]**3/(lst[1]**2*(1-lst[0])**2*lst[2])

xdata = [concated_df['Porosity'], concated_df['Surface'], concated_df['Euler_mean_vol']]
print(xdata)
ydata = concated_df['Permeability']

popt, pcov = opt.curve_fit(kozeny_carman, xdata, ydata)
# print(popt)
# print(pcov)
k = kozeny_carman(xdata, popt)
xdata_kozeny = [concated_df['Porosity'], concated_df['Surface'], k]
print(xdata_kozeny)


# make a 3d plot that plots Porosity, Surface, and Euler_mean_vol and the points are coloured to show Permeability
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol'], c=concatenated_df_rectangle['Permeability'], cmap='winter', marker='s')
# ax.scatter(concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol'], c=concatenated_df_triangle['Permeability'], cmap='winter', marker='^')
# ax.scatter(concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Euler_mean_vol'], c=concatenated_df_ellipse['Permeability'], cmap='winter', marker='o')
ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('k')
#ax.plot(xdata_kozeny[0], xdata_kozeny[1], xdata_kozeny[2], 'o')

# Prepare the data for polynomial regression
X = np.array([xdata_kozeny[0], xdata_kozeny[1]]).T
y = np.array(xdata_kozeny[2])

# Create polynomial features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Predict using the model
y_pred = model.predict(X_poly)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"RMSE: {rmse}")

# Plot the original data and the polynomial regression results
ax.scatter(xdata_kozeny[0], xdata_kozeny[1], xdata_kozeny[2], c='r', marker='o', label='Original data')
ax.scatter(xdata_kozeny[0], xdata_kozeny[1], y_pred, c='b', marker='^', label='Polynomial fit')
ax.legend()





plt.show()