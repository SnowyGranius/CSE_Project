############ SHORT GUIDE ON HOW TO USE THIS SCRIPT ############
# Chose from the folders given in lines 29-31
# Change the path in line 40 to desired dataset folder
# Run the script
# The script will output the following 3D plots:
# 1. A plot in M0, M1, M2 (colored permeability) of the data points themselves
# 2. A plot in M0, M1, M2 (colored permeability) of the data points themselves with a best-fit flat plane (a+b*M0+c*M1)
# 3. A plot in M0, M1, permeability (colored M2) of the data points with a Kozeny-Carman curve (old and new)
# 4. A plot in M0, M1, permeability (colored M2) of the data points with a Kozeny-Carman best-fit exponential (a*exp(b*M+c*M1)) and power-law (a*M0^b*M1^c) surfaces
# 5. A plot showing the evolution of k_m with porosity
# 6. A plot showing the fitted, theoretical and actual k_m values with porosity

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
from ellipse_marker import ellipse
plt.rcParams["figure.figsize"] = (16, 9)
VARIANCE=False

#Set desired SD
SD=1

# Folder options are: 
# Threshold_homogenous_diamater_wide_RCP                8000
# Threshold_homogenous_diamater_small_RCP               5000
# Porespy_homogenous_diamater                           7000

################ READING ALL THE FILES IN THE FOLDER ################
script_dir = os.path.dirname(__file__)
sub_path = 'Porespy_homogenous_diamater'
path = os.path.join(script_dir, sub_path)
variance_path=os.path.join(script_dir, 'Summaries', sub_path)
# all_files = glob.glob(os.path.join(path, "*0.300*.csv"))
all_files = glob.glob(os.path.join(path, "*rectangle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_rectangle = pd.concat(df_from_each_file, ignore_index=True)

# concatenated_df_rectangle['Permeability'] = concatenated_df_rectangle['Permeability'] * 1e5
# concatenated_df_rectangle['Energy'] = concatenated_df_rectangle['Energy'] * 1e5

#EXTRACT VARIANCE FOR RECTANGLE

variance_files = glob.glob(os.path.join(variance_path, "*rectangle*.csv"))
df_from_file = (pd.read_csv(f) for f in variance_files)
variance_df_rectangle = pd.concat(df_from_file, ignore_index=True)
variance_porosity_rectangle = variance_df_rectangle['Porosity_variance']
variance_permeability_rectangle = variance_df_rectangle['Permeability_variance']
variance_surface_rectangle = variance_df_rectangle['Surface_variance']
mean_porosity_rectangle = variance_df_rectangle['Porosity_mean']
mean_permeability_rectangle = variance_df_rectangle['Permeability_mean']
mean_surface_rectangle = variance_df_rectangle['Surface_mean']



all_files = glob.glob(os.path.join(path, "*triangle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_triangle = pd.concat(df_from_each_file, ignore_index=True)

# concatenated_df_triangle['Permeability'] = concatenated_df_triangle['Permeability'] * 1e5
# concatenated_df_triangle['Energy'] = concatenated_df_triangle['Energy'] * 1e5

#EXTRACT VARIANCE FOR TRIANGLE

variance_files = glob.glob(os.path.join(variance_path, "*triangle*.csv"))
df_from_file = (pd.read_csv(f) for f in variance_files)
variance_df_triangle = pd.concat(df_from_file, ignore_index=True)
variance_porosity_triangle = variance_df_triangle['Porosity_variance']
variance_permeability_triangle = variance_df_triangle['Permeability_variance']
variance_surface_triangle = variance_df_triangle['Surface_variance']
mean_porosity_triangle = variance_df_triangle['Porosity_mean']
mean_permeability_triangle = variance_df_triangle['Permeability_mean']
mean_surface_triangle = variance_df_triangle['Surface_mean']


all_files = glob.glob(os.path.join(path, "*ellipse*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_ellipse = pd.concat(df_from_each_file, ignore_index=True)

# concatenated_df_ellipse['Permeability'] = concatenated_df_ellipse['Permeability'] * 1e5
# concatenated_df_ellipse['Energy'] = concatenated_df_ellipse['Energy'] * 1e5

if (sub_path == 'Heterogenous_samples'):
        pass
else:
    all_files = glob.glob(os.path.join(path, "*ellipse*.csv"))
    df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
    concatenated_df_ellipse = pd.concat(df_from_each_file, ignore_index=True)

    #EXTRACT VARIANCE FOR ELLIPSE

    variance_files = glob.glob(os.path.join(variance_path, "*ellipse*.csv"))
    df_from_file = (pd.read_csv(f) for f in variance_files)
    variance_df_ellipse = pd.concat(df_from_file, ignore_index=True)
    variance_porosity_ellipse = variance_df_ellipse['Porosity_variance']
    variance_permeability_ellipse = variance_df_ellipse['Permeability_variance']
    variance_surface_ellipse = variance_df_ellipse['Surface_variance']
    mean_porosity_ellipse = variance_df_ellipse['Porosity_mean']
    mean_permeability_ellipse = variance_df_ellipse['Permeability_mean']
    mean_surface_ellipse = variance_df_ellipse['Surface_mean']

    one_sd_porosity_ellipse = variance_porosity_ellipse * mean_porosity_ellipse
    one_sd_permeability_ellipse = variance_permeability_ellipse * mean_permeability_ellipse
    one_sd_surface_ellipse = variance_surface_ellipse * mean_surface_ellipse

all_files = glob.glob(os.path.join(path, "*circle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_circle = pd.concat(df_from_each_file, ignore_index=True)

# concatenated_df_triangle['Permeability'] = concatenated_df_triangle['Permeability'] * 1e5
# concatenated_df_triangle['Energy'] = concatenated_df_triangle['Energy'] * 1e5

#EXTRACT VARIANCE FOR TRIANGLE

variance_files = glob.glob(os.path.join(variance_path, "*triangle*.csv"))
df_from_file = (pd.read_csv(f) for f in variance_files)
variance_df_circle = pd.concat(df_from_file, ignore_index=True)
variance_porosity_circle = variance_df_circle['Porosity_variance']
variance_permeability_circle = variance_df_circle['Permeability_variance']
variance_surface_circle = variance_df_circle['Surface_variance']
mean_porosity_circle = variance_df_circle['Porosity_mean']
mean_permeability_circle = variance_df_circle['Permeability_mean']
mean_surface_circle = variance_df_circle['Surface_mean']

# Calculate standard deviation (sigma) from variance
one_sd_porosity_rectangle = variance_porosity_rectangle * mean_porosity_rectangle
one_sd_permeability_rectangle = variance_permeability_rectangle * mean_permeability_rectangle
one_sd_surface_rectangle = variance_surface_rectangle * mean_surface_rectangle

one_sd_porosity_triangle = variance_porosity_triangle * mean_porosity_triangle
one_sd_permeability_triangle = variance_permeability_triangle * mean_permeability_triangle
one_sd_surface_triangle = variance_surface_triangle * mean_surface_triangle



def plot_box_and_whisker(ax, x, y, z, sigma_x, sigma_y, sigma_z, color):
        #debugging
        # print(f"Length of x: {len(x)}")
        # print(f"Length of y: {len(y)}")
        # print(f"Length of z: {len(z)}")
        # print(f"Length of sigma_x: {len(sigma_x)}")
        # print(f"Length of sigma_y: {len(sigma_y)}")
        # print(f"Length of sigma_z: {len(sigma_z)}")
        
        if VARIANCE:
            for i in range(len(x)):
                ax.plot([x[i] - SD*sigma_x[i], x[i] + SD*sigma_x[i]], [y[i], y[i]], [z[i], z[i]], color=color)
                ax.plot([x[i], x[i]], [y[i] - SD*sigma_y[i], y[i] + SD*sigma_y[i]], [z[i], z[i]], color=color)
                ax.plot([x[i], x[i]], [y[i], y[i]], [z[i] - SD*sigma_z[i], z[i] + SD*sigma_z[i]], color=color)

################ CALCULATE THE AVERAGE OF EVERY PF ################
average_samples = []
for i in range(len(concatenated_df_rectangle)):
    avg_sample = (concatenated_df_rectangle.iloc[i] + concatenated_df_triangle.iloc[i] + concatenated_df_ellipse.iloc[i] + concatenated_df_circle.iloc[i]) / 4
    average_samples.append(avg_sample)
average_samples = pd.DataFrame(average_samples)

concatenated_df_rectangle['Euler_total'] = 'Rectangle'
concatenated_df_triangle['Euler_total'] = 'Triangle'
concatenated_df_ellipse['Euler_total'] = 'Ellipse'
concatenated_df_circle['Euler_total'] = 'Circle'

concated_df = pd.concat([concatenated_df_rectangle, concatenated_df_triangle, concatenated_df_ellipse, concatenated_df_circle], ignore_index=True)


############### PLOTS OF DATA POINTS THEMSELVES ################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
all_colors = pd.concat([concatenated_df_rectangle['Permeability'], concatenated_df_triangle['Permeability'], concatenated_df_ellipse['Permeability']])
min_color = all_colors.min()
max_color = all_colors.max()
sc1 = ax.scatter(concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol'], c=concatenated_df_triangle['Permeability'], cmap='winter', marker='^', vmin=min_color, vmax=max_color)
sc2 = ax.scatter(concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol'], c=concatenated_df_rectangle['Permeability'], cmap='winter', marker='s', vmin=min_color, vmax=max_color)
sc3 = ax.scatter(concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Euler_mean_vol'], c=concatenated_df_ellipse['Permeability'], cmap='winter', marker=ellipse, vmin=min_color, vmax=max_color)
sc4 = ax.scatter(concatenated_df_circle['Porosity'], concatenated_df_circle['Surface'], concatenated_df_circle['Euler_mean_vol'], c=concatenated_df_circle['Permeability'], cmap='winter', marker='o', vmin=min_color, vmax=max_color)
cbar = plt.colorbar(sc1, ax=ax, label='Permeability')
ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('Euler Characteristic')
ax.title.set_text('4-D Plot of Porespy with Permeability as Color')
# create a folder for the plots called Porespy_plots
if not os.path.exists(os.path.join(script_dir, 'Porespy_plots')):
    os.makedirs(os.path.join(script_dir, 'Porespy_plots'))
plt.savefig(os.path.join(script_dir, 'Porespy_plots', '3d_data_points.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


################ PLOTTING A FLAT SURFACE + DATA POINTS THEMSELVES ################
x = np.linspace(min(concated_df['Porosity']), max(concated_df['Porosity']), 100)
y = np.linspace(min(concated_df['Surface']), max(concated_df['Surface']), 100)
x_grid, y_grid = np.meshgrid(x, y)

X = concated_df[['Porosity', 'Surface']]
y = concated_df['Euler_mean_vol']
linear_model = LinearRegression()
linear_model.fit(X, y)
print("M2 = ", linear_model.intercept_, " + ", linear_model.coef_[0], " M0 + ", linear_model.coef_[1], " M1")
z_grid = linear_model.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='Reds', alpha=0.7)
all_colors = pd.concat([concatenated_df_rectangle['Permeability'], concatenated_df_triangle['Permeability'], concatenated_df_ellipse['Permeability']])
min_color = all_colors.min()
max_color = all_colors.max()
sc1 = ax.scatter(concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol'], c=concatenated_df_triangle['Permeability'], cmap='winter', marker='^', vmin=min_color, vmax=max_color)
sc2 = ax.scatter(concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol'], c=concatenated_df_rectangle['Permeability'], cmap='winter', marker='s', vmin=min_color, vmax=max_color)
sc3 = ax.scatter(concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Euler_mean_vol'], c=concatenated_df_ellipse['Permeability'], cmap='winter', marker=ellipse, vmin=min_color, vmax=max_color)
sc4 = ax.scatter(concatenated_df_circle['Porosity'], concatenated_df_circle['Surface'], concatenated_df_circle['Euler_mean_vol'], c=concatenated_df_circle['Permeability'], cmap='winter', marker='o', vmin=min_color, vmax=max_color)
cbar = plt.colorbar(sc1, ax=ax, label='Permeability')
cbar.ax.set_ylabel('Permeability', rotation=90, labelpad=10, fontsize=17)
cbar.ax.tick_params(labelsize=14)  # Set colorbar text size
ax.set_xlabel('Porosity', labelpad=10, fontsize=17)
ax.set_ylabel('Surface', labelpad=10, fontsize=17)
ax.set_zlabel('Euler Characteristic', labelpad=12, fontsize=17)
ax.tick_params(axis='both', which='major', labelsize=14)  # Set axis number size
# ax.title.set_text('4-D Plot of Porespy with Euler Characteristic as Color and Exponential Surface')
# ax.title.set_fontsize(20)
#ax.set_zlim(0, 7000)
#ax.title.set_text('4-D Plot of Porespy with Permeability as Color and Best-fit Plane')
plt.savefig(os.path.join(script_dir, 'Porespy_plots', '3d_surface.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


################ DEFINITION OF KOZENY-KARMAN ################
def kozeny_carman(lst, gamma):
    return gamma*lst[0]**3/(lst[1]**2*(1-lst[0])**2*lst[2])

def kozeny_carman_plot(m0, m1, m2, gamma):
    return gamma*m0**3/(m1**2*(1-m0)**2*m2)

def kozeny_carman_new(lst, gamma):
    return lst[0]**3/(lst[1]**2*(lst[0]+lst[1]/(2*np.sqrt(lst[2])))*gamma)

def kozeny_carman_new_plot(m0, m1, m2, km):
    return m0**3/(m1**2*(m0+m1/(2*np.sqrt(m2)))*km)

def find_km(m0, m1, m2, k):
    return m0**3/(m1**2*(m0+m1/(2*np.sqrt(m2)))*k)

xdata = [concated_df['Porosity'], concated_df['Surface'], concated_df['Euler_mean_vol']]
ydata = concated_df['Permeability']


################ ORDER AND CALCULATE THE AVERAGE OF EVERY PF ################
concatenated_df_rectangle = concatenated_df_rectangle.sort_values(by='Euler_mean_vol').reset_index(drop=True)
concatenated_df_triangle = concatenated_df_triangle.sort_values(by='Euler_mean_vol').reset_index(drop=True)
concatenated_df_ellipse = concatenated_df_ellipse.sort_values(by='Euler_mean_vol').reset_index(drop=True)
concatenated_df_circle = concatenated_df_circle.sort_values(by='Euler_mean_vol').reset_index(drop=True)


xdata_rectangle = [concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol']]
xdata_triangle = [concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol']]
xdata_ellipse = [concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Euler_mean_vol']]
xdata_circle = [concatenated_df_circle['Porosity'], concatenated_df_circle['Surface'], concatenated_df_circle['Euler_mean_vol']]
# xdata_average = [average_samples['Porosity'], average_samples['Surface'], average_samples['Euler_mean_vol']]
ydata_rectangle = concatenated_df_rectangle['Permeability']
ydata_triangle = concatenated_df_triangle['Permeability']
ydata_ellipse = concatenated_df_ellipse['Permeability']
ydata_circle = concatenated_df_circle['Permeability']




# Interpolate between the samples of average_samples to have more data points
t = np.linspace(0, len(average_samples) - 1, len(average_samples))
t_interp = np.linspace(0, len(average_samples) - 1, 30)

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
xdata_ellipse_interpolated = interpolate_features(xdata_ellipse, t, t_interp)
xdata_circle_interpolated = interpolate_features(xdata_circle, t, t_interp)
ydata_rectangle_interpolated = make_interp_spline(t, ydata_rectangle, k=1)(t_interp)
ydata_triangle_interpolated = make_interp_spline(t, ydata_triangle, k=1)(t_interp)
ydata_ellipse_interpolated = make_interp_spline(t, ydata_ellipse, k=1)(t_interp)
ydata_circle_interpolated = make_interp_spline(t, ydata_circle, k=1)(t_interp)

# Calculate the average of interpolated samples
average_xdata_interp = (xdata_rectangle_interpolated + xdata_triangle_interpolated + xdata_ellipse_interpolated + xdata_circle_interpolated) / 4
average_ydata_interp = (ydata_rectangle_interpolated + ydata_triangle_interpolated + ydata_ellipse_interpolated + ydata_circle_interpolated) / 4


################ APPLYING KOZENY-CARMAN ################
average_xdata = (np.array(xdata_rectangle) + np.array(xdata_triangle) + np.array(xdata_ellipse) + np.array(xdata_circle)) / 4
average_ydata = (np.array(ydata_rectangle) + np.array(ydata_triangle) + np.array(ydata_ellipse) + np.array(ydata_circle)) / 4

popt_old, _ = opt.curve_fit(kozeny_carman, [average_xdata[0], average_xdata[1], average_xdata[2]], average_ydata)
popt_new, _ = opt.curve_fit(kozeny_carman_new, [average_xdata[0], average_xdata[1], average_xdata[2]], average_ydata)
k_old = kozeny_carman_plot(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], average_xdata_interp['Euler_mean_vol'], popt_old)
k_new = kozeny_carman_new_plot(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], average_xdata_interp['Euler_mean_vol'], popt_new)
k_new_evolution = kozeny_carman_new_plot(average_xdata[0], average_xdata[1], average_xdata[2], popt_new)
print("Best-fit k_m = ", popt_new[0])


k_star = kozeny_carman_new_plot(average_xdata[0], average_xdata[1], average_xdata[2], 2.5)
print("Theoretical k_m = 2.5")

km_list = find_km(average_xdata[0], average_xdata[1], average_xdata[2], average_ydata)
print("List of k_m values: ", km_list)
    


############### PLOTS OF KOZENY-CARMAN ################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
all_colors = pd.concat([xdata_rectangle[2], xdata_triangle[2], xdata_ellipse[2]])
min_color = all_colors.min()
max_color = all_colors.max()
ax.plot(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], k_old, label='First Kozeny-Carman Fit', color='y')
ax.plot(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], k_new, label='Improved Kozeny-Carman Fit', color='r')
sc1 = ax.scatter(xdata_rectangle[0], xdata_rectangle[1], ydata_rectangle, c=xdata_rectangle[2], marker='s', cmap='winter', vmin=min_color, vmax=max_color)
sc2 = ax.scatter(xdata_triangle[0], xdata_triangle[1], ydata_triangle, c=xdata_triangle[2], marker='^', cmap='winter', vmin=min_color, vmax=max_color)
sc3 = ax.scatter(xdata_ellipse[0], xdata_ellipse[1], ydata_ellipse, c=xdata_ellipse[2], marker=ellipse, cmap='winter', vmin=min_color, vmax=max_color)
sc4 = ax.scatter(xdata_circle[0], xdata_circle[1], ydata_circle, c=xdata_circle[2], marker='o', cmap='winter', vmin=min_color, vmax=max_color)
cbar = plt.colorbar(sc1, ax=ax, label='Euler Characteristic')

ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('Permeability')
ax.legend()
ax.view_init(elev=20, azim=205)
ax.title.set_text('4-D Plot of Porespy with Euler Characteristic as Color and Kozeny-Carman Fit')
plt.savefig(os.path.join(script_dir, 'Porespy_plots', 'kozeny_line.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()



################## EXPONENTIAL SURFACE ##################
def exponential_surface(XY, a, b, c):
    x, y = XY
    return a * np.exp(b*x) * np.exp(c*y)

def cubic_surface(XY, a, b, c, d, e, f):
    x, y = XY
    return a + b*x/y + c*(x/y)**2 + d*(x/y)**3 + e*(x/y)**4 + f*(x/y)**5

def power_law(XY, a, b, c):
    x, y = XY
    return a * x**b * y**c



# Prepare data for fitting
XY = np.vstack((concated_df['Porosity'], concated_df['Surface']))
X = concated_df['Porosity']
Y = concated_df['Surface']
Z = concated_df['Permeability']
popt_exp, _ = opt.curve_fit(exponential_surface, XY, Z, p0=(1, 0.01, 0.01))  # Initial guesses for a, b, c
popt_poly, _ = opt.curve_fit(cubic_surface, XY, Z, p0=(1, 1, 1, 1, 1, 1))  # Initial guesses for a, b, c, d, e, f
popt_power, _ = opt.curve_fit(power_law, XY, Z, p0=(1, 2, 2))  # Initial guesses for a, b, c

# Extract optimal parameters
a_exp, b_exp, c_exp = popt_exp
a_poly, b_poly, c_poly, d_poly, e_poly, f_poly = popt_poly
a_power, b_power, c_power = popt_power

print("Exponential surface: k = ", a_exp, " * exp(", b_exp, "*M0+", c_exp, "*M1)")
print("Power-law surface: k = ", a_power, " * M0^", b_power, " * M1^", c_power)

# Calculate RMSE between the surface generated and the points plotted
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# Flatten the meshgrid and calculate predicted values
predicted_values = exponential_surface((X, Y), a_exp, b_exp, c_exp)
rmse = calculate_rmse(Z, predicted_values)

print(f"RMSE between the surface generated and the points plotted: {rmse}")

# Create a meshgrid for the fitted surface
x_grid = np.linspace(X.min(), X.max(), 20)
y_grid = np.linspace(Y.min(), Y.max(), 20)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
z_mesh_exp = exponential_surface((x_mesh, y_mesh), a_exp, b_exp, c_exp)
z_mesh_poly = cubic_surface((x_mesh, y_mesh), a_poly, b_poly, c_poly, d_poly, e_poly, f_poly)
z_mesh_power = power_law((x_mesh, y_mesh), a_power, b_power, c_power)

# Plot the original points and the fitted exponential surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
all_colors = pd.concat([xdata_rectangle[2], xdata_triangle[2], xdata_ellipse[2]])
min_color = all_colors.min()
max_color = all_colors.max()
# sc = ax.scatter(X, Y, Z, c=concated_df['Euler_mean_vol'], cmap='viridis', label='Data Points')
sc1 = ax.scatter(xdata_rectangle[0], xdata_rectangle[1], ydata_rectangle, c=xdata_rectangle[2], marker='s', cmap='winter', vmin=min_color, vmax=max_color)
sc2 = ax.scatter(xdata_triangle[0], xdata_triangle[1], ydata_triangle, c=xdata_triangle[2], marker='^', cmap='winter', vmin=min_color, vmax=max_color)
sc3 = ax.scatter(xdata_ellipse[0], xdata_ellipse[1], ydata_ellipse, c=xdata_ellipse[2], marker=ellipse, cmap='winter', vmin=min_color, vmax=max_color)
sc4 = ax.scatter(xdata_circle[0], xdata_circle[1], ydata_circle, c=xdata_circle[2], marker='o', cmap='winter', vmin=min_color, vmax=max_color)
cbar = plt.colorbar(sc1, ax=ax)
cbar.ax.set_ylabel('Euler Characteristic', rotation=90, labelpad=10, fontsize=17)
cbar.ax.tick_params(labelsize=14)  # Set colorbar text size
# ax.plot_surface(x_mesh, y_mesh, z_mesh_poly, color='red', alpha=0.5, edgecolor='w', label='Fitted Polynomial Surface')
ax.plot_surface(x_mesh, y_mesh, z_mesh_exp, color='yellow', alpha=0.5, edgecolor='w', label='Fitted Exponential Surface')
#ax.plot_surface(x_mesh, y_mesh, z_mesh_power, color='green', alpha=0.5, edgecolor='w', label='Fitted Power Law Surface')
if VARIANCE:
    plot_box_and_whisker(ax, concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Permeability'], one_sd_porosity_rectangle, one_sd_surface_rectangle, one_sd_permeability_rectangle, 'blue')
    plot_box_and_whisker(ax, concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Permeability'], one_sd_porosity_triangle, one_sd_surface_triangle, one_sd_permeability_triangle, 'red')
    if (sub_path == 'Heterogenous_samples'):
        pass
    else:
        plot_box_and_whisker(ax, concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Permeability'], one_sd_porosity_ellipse, one_sd_surface_ellipse, one_sd_permeability_ellipse, 'green')

# Color bar and labels
ax.set_xlabel('Porosity', labelpad=10, fontsize=17)
ax.set_ylabel('Surface', labelpad=10, fontsize=17)
ax.set_zlabel('Permeability', fontsize=17)
ax.tick_params(axis='both', which='major', labelsize=14)  # Set axis number size
ax.view_init(elev=20, azim=135)
# ax.title.set_text('4-D Plot of Porespy with Euler Characteristic as Color and Exponential Surface')
# ax.title.set_fontsize(20)
plt.legend(fontsize=14)

# Add RMSE text to the plot
ax.text2D(0.05, 0.95, f"RMSE: {10**5*rmse:.10f}e-5", transform=ax.transAxes, fontsize=14)

plt.savefig(os.path.join(script_dir, 'Porespy_plots', 'kozeny_surface.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()



############# EVOLUTION OF KM WITH POROSITY #############
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(average_xdata[0], km_list, label='Kozeny-Carman')
ax.set_xlabel('Porosity')
ax.set_ylabel('$k_m$')
ax.grid()
plt.legend()
ax.title.set_text('Evolution of $k_m$ with Porosity')
plt.savefig(os.path.join(script_dir, 'Porespy_plots', 'km_evolution.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(average_xdata[0], average_ydata)
ax.semilogy(average_xdata[0], k_new_evolution, label='Fitted')
ax.semilogy(average_xdata[0], k_star, label='Theoretical')
ax.set_xlabel('Porosity')
ax.set_ylabel('$k$')
ax.grid()   
plt.legend()
ax.title.set_text('Fitted, Theoretical and Actual $k_m$ values with Porosity')
plt.savefig(os.path.join(script_dir, 'Porespy_plots', 'fitted_theoretical.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()