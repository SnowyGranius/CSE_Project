############ SHORT GUIDE ON HOW TO USE THIS SCRIPT ############
# File works only with Heterogenous_samples dataset
# Run the script
# The script will output the following 3D plots:
# 1. A plot in M0, M1, M2 (colored permeability) of the data points themselves
# 2. A plot in M0, M1, M2 (colored permeability) of the data points themselves with a best-fit plane
# 3. A plot in M0, M1, permeability (colored M2) of the data points with a Kozeny-Carman curve
# 4. A plot in M0, M1, permeability (colored M2) of the data points with a Kozeny-Carman best-fit surface

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
plt.rcParams["figure.figsize"] = (16, 9)

################ READING ALL THE FILES IN THE FOLDER ################
path = 'C:\\Users\\ionst\\Documents\\Fisiere_Python\\Proiect\\Data\\Datasets\\Heterogenous_samples\\'
# all_files = glob.glob(os.path.join(path, "*0.300*.csv"))
all_files = glob.glob(os.path.join(path, "*rectangle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_rectangle = pd.concat(df_from_each_file, ignore_index=True)
#print(concatenated_df_rectangle['Permeability'])
# concatenated_df_rectangle['Permeability'] = concatenated_df_rectangle['Permeability'] * 1e5
# concatenated_df_rectangle['Energy'] = concatenated_df_rectangle['Energy'] * 1e5



all_files = glob.glob(os.path.join(path, "*triangle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_triangle = pd.concat(df_from_each_file, ignore_index=True)

# concatenated_df_triangle['Permeability'] = concatenated_df_triangle['Permeability'] * 1e5
# concatenated_df_triangle['Energy'] = concatenated_df_triangle['Energy'] * 1e5


# all_files = glob.glob(os.path.join(path, "*ellipse*.csv"))
# df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
# concatenated_df_ellipse = pd.concat(df_from_each_file, ignore_index=True)

# concatenated_df_ellipse['Permeability'] = concatenated_df_ellipse['Permeability'] * 1e5
# concatenated_df_ellipse['Energy'] = concatenated_df_ellipse['Energy'] * 1e5


# detele any nan values
concatenated_df_rectangle = concatenated_df_rectangle.dropna()
concatenated_df_triangle = concatenated_df_triangle.dropna()
#concatenated_df_ellipse = concatenated_df_ellipse.dropna()

concated_df = pd.concat([concatenated_df_rectangle, concatenated_df_triangle], ignore_index=True)
#concated_df = pd.concat([concatenated_df_ellipse], ignore_index=True)




################ CALCULATE THE AVERAGE OF EVERY PF ################
average_samples = []
for i in range(len(concatenated_df_rectangle)):
    avg_sample = (concatenated_df_rectangle.iloc[i] + concatenated_df_triangle.iloc[i]) / 2
    average_samples.append(avg_sample)
average_samples = pd.DataFrame(average_samples)

# print(average_samples)
# print(concatenated_df_rectangle)
# print(concatenated_df_triangle)
# print(concatenated_df_ellipse)

concatenated_df_rectangle['Euler_total'] = 'Rectangle'
concatenated_df_triangle['Euler_total'] = 'Triangle'

concated_df = pd.concat([concatenated_df_rectangle, concatenated_df_triangle], ignore_index=True)


################ PLOTS OF DATA POINTS THEMSELVES ################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
all_colors = pd.concat([concatenated_df_rectangle['Permeability'], concatenated_df_triangle['Permeability']])
min_color = all_colors.min()
max_color = all_colors.max()
#ax.scatter(concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol'], c=concatenated_df_rectangle['Permeability'], cmap='winter', marker='s')
#ax.scatter(concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol'], c=concatenated_df_triangle['Permeability'], cmap='winter', marker='^')
#ax.scatter(concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Euler_mean_vol'], c=concatenated_df_ellipse['Permeability'], cmap='winter', marker='o')
sc1 = ax.scatter(concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol'], c=concatenated_df_triangle['Permeability'], cmap='winter', marker='^', vmin=min_color, vmax=max_color)

sc2 = ax.scatter(concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol'], c=concatenated_df_rectangle['Permeability'], cmap='winter', marker='s', vmin=min_color, vmax=max_color)

#sc3 = ax.scatter(concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Euler_mean_vol'], c=concatenated_df_ellipse['Permeability'], cmap='winter', marker='o', vmin=min_color, vmax=max_color)
#fig.colorbar(sc1, ax=ax, label='Permeability')
cbar = plt.colorbar(sc1, ax=ax, label='Permeability')

ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('Euler Mean Volume')
plt.savefig(os.path.join(path, '3d_simple.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


################ PLOTTING A FLAT SURFACE + DATA POINTS THEMSELVES ################
x = np.linspace(min(concated_df['Porosity']), max(concated_df['Porosity']), 100)
y = np.linspace(min(concated_df['Surface']), max(concated_df['Surface']), 100)
x_grid, y_grid = np.meshgrid(x, y)


X = concated_df[['Porosity', 'Surface']]
y = concated_df['Euler_mean_vol']
linear_model = LinearRegression()
linear_model.fit(X, y)
print(linear_model.coef_)


z_grid = linear_model.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='Reds', alpha=0.7)


all_colors = pd.concat([concatenated_df_rectangle['Permeability'], concatenated_df_triangle['Permeability']])
min_color = all_colors.min()
max_color = all_colors.max()
#ax.scatter(concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol'], c=concatenated_df_rectangle['Permeability'], cmap='winter', marker='s')
#ax.scatter(concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol'], c=concatenated_df_triangle['Permeability'], cmap='winter', marker='^')
#ax.scatter(concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Euler_mean_vol'], c=concatenated_df_ellipse['Permeability'], cmap='winter', marker='o')
sc1 = ax.scatter(concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol'], c=concatenated_df_triangle['Permeability'], cmap='winter', marker='^', vmin=min_color, vmax=max_color)

sc2 = ax.scatter(concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol'], c=concatenated_df_rectangle['Permeability'], cmap='winter', marker='s', vmin=min_color, vmax=max_color)

#sc3 = ax.scatter(concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Euler_mean_vol'], c=concatenated_df_ellipse['Permeability'], cmap='winter', marker='o', vmin=min_color, vmax=max_color)
#fig.colorbar(sc1, ax=ax, label='Permeability')
cbar = plt.colorbar(sc1, ax=ax, label='Permeability')

ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('Euler Mean Volume')
ax.set_zlim(0, 10000)

plt.savefig(os.path.join(path, '3d_surface.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


################ DEFINITION OF KOZENY-KARMAN ################
def kozeny_carman(lst, gamma):
    return gamma*lst[0]**3/(lst[1]**2*(1-lst[0])**2*lst[2])

def kozeny_carman_plot(m0, m1, m2, gamma):
    return gamma*m0**3/(m1**2*(1-m0)**2*m2)

xdata = [concated_df['Porosity'], concated_df['Surface'], concated_df['Euler_mean_vol']]
ydata = concated_df['Permeability']




################ ORDER AND CALCULATE THE AVERAGE OF EVERY PF ################
concatenated_df_rectangle = concatenated_df_rectangle.sort_values(by='Euler_mean_vol').reset_index(drop=True)
concatenated_df_triangle = concatenated_df_triangle.sort_values(by='Euler_mean_vol').reset_index(drop=True)
#concatenated_df_ellipse = concatenated_df_ellipse.sort_values(by='Euler_mean_vol').reset_index(drop=True)

print(concatenated_df_rectangle)
print(concatenated_df_triangle)


xdata_rectangle = [concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol']]
xdata_triangle = [concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol']]
#xdata_ellipse = [concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Euler_mean_vol']]
# xdata_average = [average_samples['Porosity'], average_samples['Surface'], average_samples['Euler_mean_vol']]
ydata_rectangle = concatenated_df_rectangle['Permeability']
ydata_triangle = concatenated_df_triangle['Permeability']
#ydata_ellipse = concatenated_df_ellipse['Permeability']




# Interpolate between the samples of average_samples to have more data points
t = np.linspace(0, len(average_samples) - 1, len(average_samples))
t_interp = np.linspace(0, len(average_samples) - 1, 30)

# Interpolating each feature
# porosity_interp = make_interp_spline(t, xdata_rectangle[0], k=1)(t_interp)
# surface_interp = make_interp_spline(t, xdata_rectangle[1], k=1)(t_interp)
# euler_mean_vol_interp = make_interp_spline(t, xdata_rectangle[2], k=1)(t_interp)

# Function to interpolate features and combine into a DataFrame
def interpolate_features(xdata, t, t_interp):
    porosity_interp = make_interp_spline(t, xdata[0], k=1)(t_interp)
    surface_interp = make_interp_spline(t, xdata[1], k=1)(t_interp)
    euler_mean_vol_interp = make_interp_spline(t, xdata[2], k=1)(t_interp)
    return pd.DataFrame({
        'Porosity': porosity_interp,
        'Surface': surface_interp,
        'Euler_mean_vol': euler_mean_vol_interp
    })

# Interpolate features for each shape
xdata_rectangle_interpolated = interpolate_features(xdata_rectangle, t, t_interp)
xdata_triangle_interpolated = interpolate_features(xdata_triangle, t, t_interp)
#xdata_ellipse_interpolated = interpolate_features(xdata_ellipse, t, t_interp)
ydata_rectangle_interpolated = make_interp_spline(t, ydata_rectangle, k=1)(t_interp)
ydata_triangle_interpolated = make_interp_spline(t, ydata_triangle, k=1)(t_interp)
#ydata_ellipse_interpolated = make_interp_spline(t, ydata_ellipse, k=1)(t_interp)

# Calculate the average of interpolated samples
average_xdata_interp = (xdata_rectangle_interpolated + xdata_triangle_interpolated) / 2
average_ydata_interp = (ydata_rectangle_interpolated + ydata_triangle_interpolated) / 2
#print(average_xdata_interp)

################ PLOTTING INTERPOLATED SAMPLES ################
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xdata_rectangle_interpolated['Porosity'], xdata_rectangle_interpolated['Surface'], xdata_rectangle_interpolated['Euler_mean_vol'], c='b', marker='s', label='Rectangle')
# ax.scatter(xdata_triangle_interpolated['Porosity'], xdata_triangle_interpolated['Surface'], xdata_triangle_interpolated['Euler_mean_vol'], c='r', marker='^', label='Triangle')
# ax.scatter(xdata_ellipse_interpolated['Porosity'], xdata_ellipse_interpolated['Surface'], xdata_ellipse_interpolated['Euler_mean_vol'], c='g', marker='o', label='Ellipse')
# ax.scatter(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], average_xdata_interp['Euler_mean_vol'], c='y', marker='o', label='Average Samples')
# ax.set_xlabel('Porosity')
# ax.set_ylabel('Surface')
# ax.set_zlabel('Euler Mean Volume')
# ax.legend()
# plt.show()




################ APPLYING KOZENY-CARMAN ################
popt, pcov = opt.curve_fit(kozeny_carman, [average_xdata_interp['Porosity'], average_xdata_interp['Surface'], average_xdata_interp['Euler_mean_vol']], average_ydata_interp)
klist = kozeny_carman_plot(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], average_xdata_interp['Euler_mean_vol'], popt)
k = kozeny_carman(xdata, popt)
k_rectangle = kozeny_carman(xdata_rectangle, popt)
k_triangle = kozeny_carman(xdata_triangle, popt)
#k_ellipse = kozeny_carman(xdata_ellipse, popt)





################ PLOTS OF KOZENY-CARMAN ################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
all_colors = pd.concat([xdata_rectangle[2], xdata_triangle[2]])
min_color = all_colors.min()
max_color = all_colors.max()
ax.plot(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], klist, label='Kozeny-Carman Fit', color='y')
sc1 = ax.scatter(xdata_rectangle[0], xdata_rectangle[1], ydata_rectangle, c=xdata_rectangle[2], marker='s', cmap='winter', vmin=min_color, vmax=max_color)
sc2 = ax.scatter(xdata_triangle[0], xdata_triangle[1], ydata_triangle, c=xdata_triangle[2], marker='^', cmap='winter', vmin=min_color, vmax=max_color)
#sc3 = ax.scatter(xdata_ellipse[0], xdata_ellipse[1], ydata_ellipse, c=xdata_ellipse[2], marker='o', cmap='winter', vmin=min_color, vmax=max_color)
cbar = plt.colorbar(sc1, ax=ax, label='Euler')

ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('Permeability')
ax.legend()
ax.view_init(elev=20, azim=205)
plt.savefig(os.path.join(path, 'kozeny_line.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()





################# KOZENY-CARMAN SURFACE PLOT ################
# Y, Z = np.meshgrid(average_xdata_interp['Porosity'], average_ydata_interp)
# X = average_xdata_interp['Surface']

# Y, Z = np.meshgrid(np.linspace(np.max(average_xdata_interp['Porosity']), np.min(average_xdata_interp['Porosity']), 100), np.linspace(np.min(average_ydata_interp), np.max([ydata_rectangle, ydata_triangle]), 100))
# X = np.linspace(np.min(average_xdata_interp['Surface']), np.max(average_xdata_interp['Surface']), 100)
# print(ydata_triangle)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(Y, X, Z, cmap='Blues', alpha=0.7)
# all_colors = pd.concat([xdata_rectangle[2], xdata_triangle[2]])
# min_color = all_colors.min()
# max_color = all_colors.max()
# # ax.plot(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], klist, label='Kozeny-Carman Fit', color='y')
# sc1 = ax.scatter(xdata_rectangle[0], xdata_rectangle[1], ydata_rectangle, c=xdata_rectangle[2], marker='s', cmap='winter', vmin=min_color, vmax=max_color)
# sc2 = ax.scatter(xdata_triangle[0], xdata_triangle[1], ydata_triangle, c=xdata_triangle[2], marker='^', cmap='winter', vmin=min_color, vmax=max_color)
# #sc3 = ax.scatter(xdata_ellipse[0], xdata_ellipse[1], ydata_ellipse, c=xdata_ellipse[2], marker='o', cmap='winter', vmin=min_color, vmax=max_color)
# cbar = plt.colorbar(sc1, ax=ax, label='Euler')
# ax.set_xlabel('Porosity')
# ax.set_ylabel('Surface')
# ax.set_zlabel('Permeability')
# ax.view_init(elev=20, azim=205)
# plt.savefig(os.path.join(path, 'kozeny_surface.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()


################## EXPONENTIAL SURFACE ##################
def exponential_surface(XY, a, b, c):
    x, y = XY
    return a * np.exp(b*x) * np.exp(c*y)

def cubic_surface(XY, a, b, c, d, e):
    x, y = XY
    return a + b*x + c*y + d*x**2 + e*y**2

# Prepare data for fitting
XY = np.vstack((concated_df['Porosity'], concated_df['Surface']))
X = concated_df['Porosity']
Y = concated_df['Surface']
Z = concated_df['Permeability']
popt_exp, _ = opt.curve_fit(exponential_surface, XY, Z, p0=(1, 0.01, 0.01))  # Initial guesses for a, b, c
popt_poly, _ = opt.curve_fit(cubic_surface, XY, Z, p0=(1, 1, 1, 1, 1))  # Initial guesses for a, b, c, d, e, f

# Extract optimal parameters
a, b, c = popt_exp
# a, b, c, d, e = popt_poly
print(popt_exp)

# Create a meshgrid for the fitted surface
x_grid = np.linspace(X.min(), X.max(), 30)
y_grid = np.linspace(Y.min(), Y.max(), 30)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
z_mesh_exp = exponential_surface((x_mesh, y_mesh), a, b, c)
# z_mesh_poly = cubic_surface((x_mesh, y_mesh), a, b, c, e, f)

# Plot the original points and the fitted exponential surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
all_colors = pd.concat([xdata_rectangle[2], xdata_triangle[2]])
min_color = all_colors.min()
max_color = all_colors.max()
# sc = ax.scatter(X, Y, Z, c=concated_df['Euler_mean_vol'], cmap='viridis', label='Data Points')
sc1 = ax.scatter(xdata_rectangle[0], xdata_rectangle[1], ydata_rectangle, c=xdata_rectangle[2], marker='s', cmap='winter', vmin=min_color, vmax=max_color)
sc2 = ax.scatter(xdata_triangle[0], xdata_triangle[1], ydata_triangle, c=xdata_triangle[2], marker='^', cmap='winter', vmin=min_color, vmax=max_color)
cbar = plt.colorbar(sc1, ax=ax, label='Euler')
ax.plot_surface(x_mesh, y_mesh, z_mesh_exp, color='yellow', alpha=0.5, edgecolor='w', label='Fitted Exponential Surface')

# Color bar and labels
ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('Permeability')
ax.view_init(elev=20, azim=135)
plt.legend()
plt.savefig(os.path.join(path, 'kozeny_surface.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()