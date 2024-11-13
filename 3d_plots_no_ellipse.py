############ SHORT GUIDE ON HOW TO USE THIS SCRIPT ############
# File works only with Heterogenous_samples dataset
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
plt.rcParams["figure.figsize"] = (16, 9)
VARIANCE=False

################ READING ALL THE FILES IN THE FOLDER ################
script_dir = os.path.dirname(__file__)
sub_path = 'Heterogenous_samples'
path = os.path.join(script_dir, sub_path)
variance_path=os.path.join(script_dir, 'Summaries', sub_path)
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

    sigma_porosity_ellipse = np.sqrt(variance_porosity_ellipse) * mean_porosity_ellipse
    sigma_permeability_ellipse = np.sqrt(variance_permeability_ellipse) * mean_permeability_ellipse
    sigma_surface_ellipse = np.sqrt(variance_surface_ellipse) * mean_surface_ellipse

 # Calculate standard deviation (sigma) from variance
sigma_porosity_rectangle = np.sqrt(variance_porosity_rectangle) * mean_porosity_rectangle
sigma_permeability_rectangle = np.sqrt(variance_permeability_rectangle) * mean_permeability_rectangle
sigma_surface_rectangle = np.sqrt(variance_surface_rectangle) * mean_surface_rectangle

sigma_porosity_triangle = np.sqrt(variance_porosity_triangle) * mean_porosity_triangle
sigma_permeability_triangle = np.sqrt(variance_permeability_triangle) * mean_permeability_triangle
sigma_surface_triangle = np.sqrt(variance_surface_triangle) * mean_surface_triangle


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
                ax.plot([x[i] - sigma_x[i], x[i] + sigma_x[i]], [y[i], y[i]], [z[i], z[i]], color=color)
                ax.plot([x[i], x[i]], [y[i] - sigma_y[i], y[i] + sigma_y[i]], [z[i], z[i]], color=color)
                ax.plot([x[i], x[i]], [y[i], y[i]], [z[i] - sigma_z[i], z[i] + sigma_z[i]], color=color)

################ CALCULATE THE AVERAGE OF EVERY PF ################
average_samples = []
for i in range(len(concatenated_df_rectangle)):
    avg_sample = (concatenated_df_rectangle.iloc[i] + concatenated_df_triangle.iloc[i]) / 2
    average_samples.append(avg_sample)
average_samples = pd.DataFrame(average_samples)


concatenated_df_rectangle['Euler_total'] = 'Rectangle'
concatenated_df_triangle['Euler_total'] = 'Triangle'

concated_df = pd.concat([concatenated_df_rectangle, concatenated_df_triangle], ignore_index=True)


############### PLOTS OF DATA POINTS THEMSELVES ################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
all_colors = pd.concat([concatenated_df_rectangle['Permeability'], concatenated_df_triangle['Permeability']])
min_color = all_colors.min()
max_color = all_colors.max()
sc1 = ax.scatter(concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol'], c=concatenated_df_triangle['Permeability'], cmap='winter', marker='^', vmin=min_color, vmax=max_color)
sc2 = ax.scatter(concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol'], c=concatenated_df_rectangle['Permeability'], cmap='winter', marker='s', vmin=min_color, vmax=max_color)
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
print("M2 = ", linear_model.intercept_, " + ", linear_model.coef_[0], " M0 + ", linear_model.coef_[1], " M1")
z_grid = linear_model.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='Reds', alpha=0.7)
all_colors = pd.concat([concatenated_df_rectangle['Permeability'], concatenated_df_triangle['Permeability']])
min_color = all_colors.min()
max_color = all_colors.max()
sc1 = ax.scatter(concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol'], c=concatenated_df_triangle['Permeability'], cmap='winter', marker='^', vmin=min_color, vmax=max_color)
sc2 = ax.scatter(concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol'], c=concatenated_df_rectangle['Permeability'], cmap='winter', marker='s', vmin=min_color, vmax=max_color)
cbar = plt.colorbar(sc1, ax=ax, label='Permeability')
ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('Euler Mean Volume')
ax.set_zlim(0, 7000)
plt.savefig(os.path.join(path, '3d_surface.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
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


xdata_rectangle = [concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Euler_mean_vol']]
xdata_triangle = [concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Euler_mean_vol']]
ydata_rectangle = concatenated_df_rectangle['Permeability']
ydata_triangle = concatenated_df_triangle['Permeability']




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
ydata_rectangle_interpolated = make_interp_spline(t, ydata_rectangle, k=1)(t_interp)
ydata_triangle_interpolated = make_interp_spline(t, ydata_triangle, k=1)(t_interp)

# Calculate the average of interpolated samples
average_xdata_interp = (xdata_rectangle_interpolated + xdata_triangle_interpolated) / 2
average_ydata_interp = (ydata_rectangle_interpolated + ydata_triangle_interpolated) / 2






################ APPLYING KOZENY-CARMAN ################
average_xdata = (np.array(xdata_rectangle) + np.array(xdata_triangle)) / 2
average_ydata = (np.array(ydata_rectangle) + np.array(ydata_triangle)) / 2

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
all_colors = pd.concat([xdata_rectangle[2], xdata_triangle[2]])
min_color = all_colors.min()
max_color = all_colors.max()
ax.plot(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], k_old, label='Kozeny-Carman Fit', color='y')
ax.plot(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], k_new, label='New Kozeny-Carman Fit', color='r')
sc1 = ax.scatter(xdata_rectangle[0], xdata_rectangle[1], ydata_rectangle, c=xdata_rectangle[2], marker='s', cmap='winter', vmin=min_color, vmax=max_color)
sc2 = ax.scatter(xdata_triangle[0], xdata_triangle[1], ydata_triangle, c=xdata_triangle[2], marker='^', cmap='winter', vmin=min_color, vmax=max_color)
cbar = plt.colorbar(sc1, ax=ax, label='Euler')

ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('Permeability')
ax.legend()
ax.view_init(elev=20, azim=205)
plt.savefig(os.path.join(path, 'kozeny_line.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
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
all_colors = pd.concat([xdata_rectangle[2], xdata_triangle[2]])
min_color = all_colors.min()
max_color = all_colors.max()
sc1 = ax.scatter(xdata_rectangle[0], xdata_rectangle[1], ydata_rectangle, c=xdata_rectangle[2], marker='s', cmap='winter', vmin=min_color, vmax=max_color)
sc2 = ax.scatter(xdata_triangle[0], xdata_triangle[1], ydata_triangle, c=xdata_triangle[2], marker='^', cmap='winter', vmin=min_color, vmax=max_color)
cbar = plt.colorbar(sc1, ax=ax, label='Euler')
ax.plot_surface(x_mesh, y_mesh, z_mesh_exp, color='yellow', alpha=0.5, edgecolor='w', label='Fitted Exponential Surface')
ax.plot_surface(x_mesh, y_mesh, z_mesh_power, color='green', alpha=0.5, edgecolor='w', label='Fitted Power Law Surface')
if VARIANCE:
        plot_box_and_whisker(ax, concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Permeability'], sigma_porosity_rectangle, sigma_surface_rectangle, sigma_permeability_rectangle, 'blue')
        plot_box_and_whisker(ax, concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Permeability'], sigma_porosity_triangle, sigma_surface_triangle, sigma_permeability_triangle, 'red')
        if (sub_path == 'Heterogenous_samples'):
            pass
        else:
            plot_box_and_whisker(ax, concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Permeability'], sigma_porosity_ellipse, sigma_surface_ellipse, sigma_permeability_ellipse, 'green')

# Color bar and labels
ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('Permeability')
ax.view_init(elev=20, azim=135)
plt.legend()

# Add RMSE text to the plot
ax.text2D(0.05, 0.95, f"RMSE: {10**5*rmse:.10f}e-5", transform=ax.transAxes)

plt.savefig(os.path.join(path, 'kozeny_surface.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()



############# EVOLUTION OF KM WITH POROSITY #############
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(average_xdata[0], km_list, label='Kozeny-Carman')
ax.set_xlabel('Porosity')
ax.set_ylabel('k_m')
ax.grid()
plt.legend()
plt.savefig(os.path.join(path, 'km_evolution.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(average_xdata[0], average_ydata)
ax.semilogy(average_xdata[0], k_new_evolution, label='fitted')
ax.semilogy(average_xdata[0], k_star, label='theoretical')
ax.set_xlabel('Porosity')
ax.set_ylabel('k')
ax.grid()
plt.legend()
plt.savefig(os.path.join(path, 'fitted_theoretical.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()




