import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.interpolate import griddata
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load the CSV file
# csv_file_path = 'Microstructure_generation\microstructure_data.csv'
script_dir = os.path.dirname(__file__)
base_path=script_dir
path=os.path.join(base_path, 'Microstructure_generation', 'Initial_data', 'microstructure_data.csv')
data = pd.read_csv(path)

# Extract M0, M1, and M2 columns
M0 = data['M0']
M1 = data['M1']
M2 = data['M2']
seed=data['Seed']


###--------------DATA FILTERING---------------###
# Filter data for the first 1 seeds
filtered_data = data[data['Seed'].isin(data['Seed'].unique()[:10])]  #CHANGE THE SLICING TO INCLUDE ALL POINTS


# Create a 3D plot with all the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract M0, M1, and M2 columns for the filtered data
M0_filtered = filtered_data['M0']
M1_filtered = filtered_data['M1']
M2_filtered = filtered_data['M2']
seed_filtered = filtered_data['Seed']

# Plot the filtered data
scatter = ax.scatter(M0_filtered, M1_filtered, M2_filtered, c=seed_filtered, marker='o')
ax.set_xlabel('M0')
ax.set_ylabel('M1')
ax.set_zlabel('M2')
ax.set_title('3D Plot of M0, M1, and M2 for the first 5 seeds')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Seed')
plt.show()



# -----------------SURFACE FITTING-----------------#
# Create grid data for surface fitting
grid_x, grid_y = np.mgrid[M0_filtered.min():M0_filtered.max():100j, M1_filtered.min():M1_filtered.max():100j]

# Interpolate the data
grid_z = griddata((M0_filtered, M1_filtered), M2_filtered, (grid_x, grid_y), method='cubic')

# Define the mathematical expression for the surface
def surface_function(x, y, a, b, c, d):
    return a * x ** 3 + b * y + d * x ** 2 + c * x ** 2 * y

# Flatten the grid data for curve fitting
x_data = np.vstack((M0_filtered, M1_filtered))
y_data = M2_filtered

# Perform curve fitting to find the optimal coefficients
popt, pcov = curve_fit(lambda xy, a, b, c, d: surface_function(xy[0], xy[1], a, b, c, d), x_data, y_data)

# Extract the optimal coefficients
a_opt, b_opt, c_opt, d_opt = popt

# Apply the optimized surface function to the grid data
grid_z = surface_function(grid_x, grid_y, a_opt, b_opt, c_opt, d_opt)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.6)

# Plot the original data points
modified_plot=ax.scatter(M0, M1, M2, c=seed, marker='o')


# Set labels
ax.set_xlabel('M0')
ax.set_ylabel('M1')
ax.set_zlabel('M2')
ax.set_title('3D Surface Plot of M0, M1, and M2')
cbar = plt.colorbar(modified_plot, ax=ax)
cbar.set_label('Seed')

# Show the plot
plt.show()




# Create many 3D plots and iterate through the blobiness values
for i in range (1, 11):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Filter data for a specific blobiness value
    blobiness_value = i  # Change this to the desired blobiness value
    filtered_data = data[data['Blobiness'] == blobiness_value]

    # Extract M0, M1, and M2 columns for the filtered data
    M0_filtered = filtered_data['M0']
    M1_filtered = filtered_data['M1']
    M2_filtered = filtered_data['M2']
    seed_filtered = filtered_data['Seed']

    # Plot the filtered data
    scatter = ax.scatter(M0_filtered, M1_filtered, M2_filtered, c=seed_filtered, marker='o')

    # Set labels
    ax.set_xlabel('M0')
    ax.set_ylabel('M1')
    ax.set_zlabel('M2')

    #set maximum bounds
    ax.set_xlim([data['M0'].min(), data['M0'].max()])
    ax.set_ylim([data['M1'].min(), data['M1'].max()])
    ax.set_zlim([data['M2'].min(), data['M2'].max()])

    # Set title
    ax.set_title(f'3D Plot of M0, M1, and M2 for Blobiness {blobiness_value}')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Seed')

    # Show the plot
    plt.show()
    