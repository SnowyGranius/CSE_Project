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

path = 'C:\\Users\\ionst\\Documents\\Fisiere_Python\\Proiect\\Data\\Datasets\\Threshold_homogenous_diamater_small_RCP\\'
# all_files = glob.glob(os.path.join(path, "*0.300*.csv"))
all_files = glob.glob(os.path.join(path, "*rectangle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_rectangle = pd.concat(df_from_each_file, ignore_index=True)
# concatenated_df_rectangle['Permeability'] = concatenated_df_rectangle['Permeability'] * 1e5
# concatenated_df_rectangle['Energy'] = concatenated_df_rectangle['Energy'] * 1e5


all_files = glob.glob(os.path.join(path, "*triangle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_triangle = pd.concat(df_from_each_file, ignore_index=True)

# concatenated_df_triangle['Permeability'] = concatenated_df_triangle['Permeability'] * 1e5
# concatenated_df_triangle['Energy'] = concatenated_df_triangle['Energy'] * 1e5


all_files = glob.glob(os.path.join(path, "*ellipse*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_ellipse = pd.concat(df_from_each_file, ignore_index=True)

# concatenated_df_ellipse['Permeability'] = concatenated_df_ellipse['Permeability'] * 1e5
# concatenated_df_ellipse['Energy'] = concatenated_df_ellipse['Energy'] * 1e5




# concated_df = pd.concat([concatenated_df_rectangle, concatenated_df_triangle], ignore_index=True)
# concated_df = pd.concat([concatenated_df_ellipse], ignore_index=True)
#print(concated_df)

# Calculate the average for each sample between rectangle, ellipse, and triangle

average_samples = []
for i in range(len(concatenated_df_rectangle)):
    avg_sample = (concatenated_df_rectangle.iloc[i] + concatenated_df_triangle.iloc[i] + concatenated_df_ellipse.iloc[i]) / 3
    average_samples.append(avg_sample)
average_samples = pd.DataFrame(average_samples)

# print(average_samples)
# print(concatenated_df_rectangle)
# print(concatenated_df_triangle)
# print(concatenated_df_ellipse)

concatenated_df_rectangle['Euler_total'] = 'Rectangle'
concatenated_df_triangle['Euler_total'] = 'Triangle'
concatenated_df_ellipse['Euler_total'] = 'Ellipse'

concated_df = pd.concat([concatenated_df_rectangle, concatenated_df_triangle, concatenated_df_ellipse], ignore_index=True)


def kozeny_carman(lst, gamma):
    return gamma*lst[0]**3/(lst[1]**2*(1-lst[0])**2*lst[2])

def kozeny_carman_plot(m0, m1, m2, gamma):
    return gamma*m0**3/(m1**2*(1-m0)**2*m2)

xdata = [concated_df['Porosity'], concated_df['Surface'], concated_df['Euler_mean_vol']]
ydata = concated_df['Permeability']

xdata_rectangle = [concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol']]
xdata_triangle = [concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol']]
xdata_ellipse = [concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Euler_mean_vol']]
# xdata_average = [average_samples['Porosity'], average_samples['Surface'], average_samples['Euler_mean_vol']]
ydata_rectangle = concatenated_df_rectangle['Permeability']
ydata_triangle = concatenated_df_triangle['Permeability']
ydata_ellipse = concatenated_df_ellipse['Permeability']





# Interpolate between the samples of average_samples to have more data points
t = np.linspace(0, len(average_samples) - 1, len(average_samples))
t_interp = np.linspace(0, len(average_samples) - 1, 100)

# Interpolating each feature
porosity_interp = make_interp_spline(t, xdata_rectangle[0], k=2)(t_interp)
surface_interp = make_interp_spline(t, xdata_rectangle[1], k=2)(t_interp)
euler_mean_vol_interp = make_interp_spline(t, xdata_rectangle[2], k=2)(t_interp)

# Function to interpolate features and combine into a DataFrame
def interpolate_features(xdata, t, t_interp):
    porosity_interp = make_interp_spline(t, xdata[0], k=2)(t_interp)
    surface_interp = make_interp_spline(t, xdata[1], k=2)(t_interp)
    euler_mean_vol_interp = make_interp_spline(t, xdata[2], k=2)(t_interp)
    return pd.DataFrame({
        'Porosity': porosity_interp,
        'Surface': surface_interp,
        'Euler_mean_vol': euler_mean_vol_interp
    })

# Interpolate features for each shape
xdata_rectangle_interpolated = interpolate_features(xdata_rectangle, t, t_interp)
xdata_triangle_interpolated = interpolate_features(xdata_triangle, t, t_interp)
xdata_ellipse_interpolated = interpolate_features(xdata_ellipse, t, t_interp)
ydata_rectangle_interpolated = make_interp_spline(t, ydata_rectangle, k=2)(t_interp)
ydata_triangle_interpolated = make_interp_spline(t, ydata_triangle, k=2)(t_interp)
ydata_ellipse_interpolated = make_interp_spline(t, ydata_ellipse, k=2)(t_interp)

# Calculate the average of interpolated samples
average_xdata_interp = (xdata_rectangle_interpolated + xdata_triangle_interpolated + xdata_ellipse_interpolated) / 3
average_ydata_interp = (ydata_rectangle_interpolated + ydata_triangle_interpolated + ydata_ellipse_interpolated) / 3
#print(average_xdata_interp)

# Plot the interpolated samples
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(interpolated_samples['Porosity'], interpolated_samples['Surface'], interpolated_samples['Euler_mean_vol'], c='r', marker='o', label='Interpolated Samples')
ax.scatter(xdata_rectangle_interpolated['Porosity'], xdata_rectangle_interpolated['Surface'], xdata_rectangle_interpolated['Euler_mean_vol'], c='b', marker='s', label='Rectangle')
ax.scatter(xdata_triangle_interpolated['Porosity'], xdata_triangle_interpolated['Surface'], xdata_triangle_interpolated['Euler_mean_vol'], c='r', marker='^', label='Triangle')
ax.scatter(xdata_ellipse_interpolated['Porosity'], xdata_ellipse_interpolated['Surface'], xdata_ellipse_interpolated['Euler_mean_vol'], c='g', marker='o', label='Ellipse')
ax.scatter(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], average_xdata_interp['Euler_mean_vol'], c='y', marker='o', label='Average Samples')
ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('Euler Mean Volume')
ax.legend()
plt.show()

# # Interpolate the data set ydata and make 100 points
# x_interp = np.linspace(0, len(xdata[2]) - 1, 100)
# #y_interp = np.interp(x_interp, np.arange(len(ydata)), ydata)
# y_interp = make_interp_spline(np.arange(len(xdata[2])), xdata[2], k=2)(x_interp)
# print(y_interp)
# # xdata_interp = [make_interp_spline(np.arange(len(x)), x, k=3)(x_interp) for x in xdata]
# # print(xdata_interp)
# x1 = np.linspace(np.max(xdata[0]), np.min(xdata[0]), 100)
# x1_interp = make_interp_spline(np.arange(len(xdata[0])), xdata[0], k=2)(x_interp)
# x2 = np.linspace(np.min(xdata[1]), np.max(xdata[1]), 100)
# x2_interp = make_interp_spline(np.arange(len(xdata[1])), xdata[1], k=2)(x_interp)
# print(x1, x2)
# plt.show()


#print(average_xdata_interp)

popt, pcov = opt.curve_fit(kozeny_carman, [average_xdata_interp['Porosity'], average_xdata_interp['Surface'], average_xdata_interp['Euler_mean_vol']], average_ydata_interp)
klist = kozeny_carman_plot(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], average_xdata_interp['Euler_mean_vol'], popt)
# print(klist)
# # print(popt)
# # print(pcov)
k = kozeny_carman(xdata, popt)
k_rectangle = kozeny_carman(xdata_rectangle, popt)
k_triangle = kozeny_carman(xdata_triangle, popt)
k_ellipse = kozeny_carman(xdata_ellipse, popt)
print(k)
print(xdata)
# print(k)
# xdata_kozeny = [concated_df['Porosity'], concated_df['Surface'], k]
# #print(xdata_kozeny)


# make a 3d plot that plots Porosity, Surface, and Euler_mean_vol and the points are coloured to show Permeability
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol'], c=concatenated_df_rectangle['Permeability'], cmap='winter', marker='s')
ax.scatter(concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol'], c=concatenated_df_triangle['Permeability'], cmap='winter', marker='^')
ax.scatter(concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Euler_mean_vol'], c=concatenated_df_ellipse['Permeability'], cmap='winter', marker='o')
plt.show()
# Plot the Kozeny-Carman fit
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], klist, label='Kozeny-Carman Fit', color='y')
ax.scatter(xdata_rectangle[0], xdata_rectangle[1], k_rectangle, marker='s')
ax.scatter(xdata_triangle[0], xdata_triangle[1], k_triangle, marker='^')
ax.scatter(xdata_ellipse[0], xdata_ellipse[1], k_ellipse, marker='o')

# Add labels and legend
ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('Permeability')
ax.legend()

#ax.scatter(xdata_average[0], xdata_average[1], xdata_average[2], c='r', marker='o')
#ax.plot(xdata_interp[0], xdata_interp[1], xdata_interp[2], label='Interpolated Data', color='r')
# ax.set_xlabel('Porosity')
# ax.set_ylabel('Surface')
# ax.set_zlabel('k')
# ax.plot(xdata_kozeny[0], xdata_kozeny[1], xdata_kozeny[2], 'o')
# ax.plot(x1, x2, klist, 'r')

# Prepare the data for polynomial regression
# X = np.array([xdata_kozeny[0], xdata_kozeny[1]]).T
# y = np.array(xdata_kozeny[2])

# # Create polynomial features
# poly = PolynomialFeatures(degree=3)
# X_poly = poly.fit_transform(X)

# # Fit the polynomial regression model
# model = LinearRegression()
# model.fit(X_poly, y)

# # Predict using the model
# y_pred = model.predict(X_poly)

# # Calculate RMSE
# rmse = np.sqrt(mean_squared_error(y, y_pred))
# print(f"RMSE: {rmse}")

# # Plot the original data and the polynomial regression results
# ax.scatter(xdata_kozeny[0], xdata_kozeny[1], xdata_kozeny[2], c='r', marker='o', label='Original data')
# ax.scatter(xdata_kozeny[0], xdata_kozeny[1], y_pred, c='b', marker='^', label='Polynomial fit')
# ax.legend()





plt.show()