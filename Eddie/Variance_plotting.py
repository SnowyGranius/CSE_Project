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


#THIS PROGRAM IS USED TO PLOT THE BOX AND WHISKER PLOTS FOR ALL THE SAMPLES

#set this to 0 to just plot the data
VARIANCE=1

#set this to 3 or 1 depending on the SD you want to plot
SD=3


# # Read the data from the csv file
# df = pd.read_csv('C:\\Users\\ionst\\Documents\\Fisiere Python\\Proiect\\Data\Datasets\\Heterogenous_samples\\Minkowskis_pf_0.220_rectangle_1_hetero.csv')
# print(df)
script_dir = os.path.dirname(__file__)

# Read all relevant files in the folder
path_list=['Heterogenous_samples', 'Threshold_homogenous_diameter_small_RCP', 'Threshold_homogenous_diameter_wide_RCP', 'Porespy_homogenous_diameter']
base_path=script_dir  #'E:\TU Delft\BSc 3\CSE Minor\TW3715TU Project A\Minkowskis_project\Eddie'
for sub_path in path_list:
    path=os.path.join(base_path, sub_path)
    print(path)
    #path = 'E:\TU Delft\BSc 3\CSE Minor\TW3715TU Project A\Minkowskis_project\Datasets\Porespy_homogenous_diamater'
    variance_path=os.path.join(base_path, 'Summaries', sub_path)
    print(variance_path)
    # all_files = glob.glob(os.path.join(path, "*0.300*.csv"))
    all_files = glob.glob(os.path.join(path, "*rectangle*.csv"))
    df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
    concatenated_df_rectangle = pd.concat(df_from_each_file, ignore_index=True)

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


    # concatenated_df_rectangle['Permeability'] = concatenated_df_rectangle['Permeability'] * 1e5
    # concatenated_df_rectangle['Energy'] = concatenated_df_rectangle['Energy'] * 1e5


    all_files = glob.glob(os.path.join(path, "*triangle*.csv"))
    df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
    concatenated_df_triangle = pd.concat(df_from_each_file, ignore_index=True)

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
    #print(variance_porosity_triangle)
    #print(variance_df_triangle*mean_porosity_triangle)


    # concatenated_df_triangle['Permeability'] = concatenated_df_triangle['Permeability'] * 1e5
    # concatenated_df_triangle['Energy'] = concatenated_df_triangle['Energy'] * 1e5

    #filename_to_check=os.path.join(path, "*ellipse*.csv")
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

        print(variance_porosity_ellipse*mean_porosity_ellipse)
        print(variance_permeability_ellipse*mean_permeability_ellipse)
        print(variance_surface_ellipse*mean_surface_ellipse)

        one_sd_porosity_ellipse = variance_porosity_ellipse * mean_porosity_ellipse
        one_sd_permeability_ellipse = variance_permeability_ellipse * mean_permeability_ellipse
        one_sd_surface_ellipse = variance_surface_ellipse * mean_surface_ellipse

    # concatenated_df_ellipse['Permeability'] = concatenated_df_ellipse['Permeability'] * 1e5
    # concatenated_df_ellipse['Energy'] = concatenated_df_ellipse['Energy'] * 1e5

    # Calculate standard deviation (sigma) from variance
    one_sd_porosity_rectangle = variance_porosity_rectangle * mean_porosity_rectangle
    one_sd_permeability_rectangle = variance_permeability_rectangle * mean_permeability_rectangle
    one_sd_surface_rectangle = variance_surface_rectangle * mean_surface_rectangle

    one_sd_porosity_triangle = variance_porosity_triangle * mean_porosity_triangle
    one_sd_permeability_triangle = variance_permeability_triangle * mean_permeability_triangle
    one_sd_surface_triangle = variance_surface_triangle * mean_surface_triangle

    # print(sigma_porosity_rectangle)
    # print(sigma_permeability_rectangle)
    # print(sigma_surface_rectangle)

    # Function to plot box and whisker plots on a 3D graph
    def plot_box_and_whisker(ax, x, y, z, sigma_x, sigma_y, sigma_z, color):
        #debugging
        # print(f"Length of x: {len(x)}")
        # print(f"Length of y: {len(y)}")
        # print(f"Length of z: {len(z)}")
        # print(f"Length of sigma_x: {len(sigma_x)}")
        # print(f"Length of sigma_y: {len(sigma_y)}")
        # print(f"Length of sigma_z: {len(sigma_z)}")
        
        if VARIANCE==1:
            for i in range(len(x)):
                ax.plot([x[i] - SD*sigma_x[i], x[i] + SD*sigma_x[i]], [y[i], y[i]], [z[i], z[i]], color=color)
                ax.plot([x[i], x[i]], [y[i] - SD*sigma_y[i], y[i] + SD*sigma_y[i]], [z[i], z[i]], color=color)
                ax.plot([x[i], x[i]], [y[i], y[i]], [z[i] - SD*sigma_z[i], z[i] + SD*sigma_z[i]], color=color)

    # Plot the 3D graph with box and whisker plots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points
    sc1=ax.scatter(concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Permeability'], c=concatenated_df_rectangle['Euler_mean_vol'], cmap='winter', marker='s')
    sc2=ax.scatter(concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Permeability'], c=concatenated_df_triangle['Euler_mean_vol'], cmap='winter', marker='^')
    if (sub_path == 'Heterogenous_samples'):
        pass
    else:
        sc3=ax.scatter(concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Permeability'], c=concatenated_df_ellipse['Euler_mean_vol'], cmap='winter', marker='o')

    # Plot box and whisker plots
    if VARIANCE==1:
        plot_box_and_whisker(ax, concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Permeability'], one_sd_porosity_rectangle, one_sd_surface_rectangle, one_sd_permeability_rectangle, 'blue')
        plot_box_and_whisker(ax, concatenated_df_triangle['Porosity'], concatenated_df_triangle['Surface'], concatenated_df_triangle['Permeability'], one_sd_porosity_triangle, one_sd_surface_triangle, one_sd_permeability_triangle, 'red')
        if (sub_path == 'Heterogenous_samples'):
            # Determine the min and max values for Porosity, Surface, and Permeability across all shapes
            min_porosity = min(concatenated_df_rectangle['Porosity'].min(), concatenated_df_triangle['Porosity'].min())
            max_porosity = max(concatenated_df_rectangle['Porosity'].max(), concatenated_df_triangle['Porosity'].max())
            
            min_surface = min(concatenated_df_rectangle['Surface'].min(), concatenated_df_triangle['Surface'].min())
            max_surface = max(concatenated_df_rectangle['Surface'].max(), concatenated_df_triangle['Surface'].max())
            
            min_permeability = min(concatenated_df_rectangle['Permeability'].min(), concatenated_df_triangle['Permeability'].min())
            max_permeability = max(concatenated_df_rectangle['Permeability'].max(), concatenated_df_triangle['Permeability'].max())

            pass
        else:
            plot_box_and_whisker(ax, concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Permeability'], one_sd_porosity_ellipse, one_sd_surface_ellipse, one_sd_permeability_ellipse, 'green')
            min_porosity = min(concatenated_df_rectangle['Porosity'].min(), concatenated_df_triangle['Porosity'].min(), concatenated_df_ellipse['Porosity'].min())
            max_porosity = max(concatenated_df_rectangle['Porosity'].max(), concatenated_df_triangle['Porosity'].max(), concatenated_df_ellipse['Porosity'].max())
            
            min_surface = min(concatenated_df_rectangle['Surface'].min(), concatenated_df_triangle['Surface'].min(), concatenated_df_ellipse['Surface'].min())
            max_surface = max(concatenated_df_rectangle['Surface'].max(), concatenated_df_triangle['Surface'].max(), concatenated_df_ellipse['Surface'].max())
            
            min_permeability = min(concatenated_df_rectangle['Permeability'].min(), concatenated_df_triangle['Permeability'].min(), concatenated_df_ellipse['Permeability'].min())
            max_permeability = max(concatenated_df_rectangle['Permeability'].max(), concatenated_df_triangle['Permeability'].max(), concatenated_df_ellipse['Permeability'].max())


    # Set axis labels
    ax.set_xlabel('Porosity')
    ax.set_ylabel('Surface')
    ax.set_zlabel('Permeability')

    ax.set_title('Box and Whisker Plots for ' + sub_path.replace('_', ' '))
    #set log scale permeability and set limits
    ax.set_zscale('log')

    # Set axis limits based on the min and max values
    ax.set_xlim([min_porosity, max_porosity])
    ax.set_ylim([min_surface, max_surface])
    ax.set_zlim([min_permeability, max_permeability])
    ax.view_init(elev=25, azim=160)  # Adjust the elevation and azimuthal angles as needed

    # Add color bar
    cbar = plt.colorbar(sc1, ax=ax, pad=0.1)
    cbar.set_label('Euler Mean Volume')

    # Show the plot
    plt.show()


'''


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


# Set axis limits
ax.set_xlim([concatenated_df_rectangle['Porosity'].min(), concatenated_df_rectangle['Porosity'].max()])
ax.set_ylim([concatenated_df_rectangle['Surface'].min(), concatenated_df_rectangle['Surface'].max()])
ax.set_zlim([concatenated_df_rectangle['Euler_mean_vol'].min(), concatenated_df_rectangle['Euler_mean_vol'].max()])

ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('Permeability')
#ax.set_zscale('log')

plt.show()

# Plot the Kozeny-Carman fit
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(average_xdata_interp['Porosity'], average_xdata_interp['Surface'], klist, label='Kozeny-Carman Fit', color='y')
ax.scatter(xdata_rectangle[0], xdata_rectangle[1], k_rectangle, marker='s')
ax.scatter(xdata_triangle[0], xdata_triangle[1], k_triangle, marker='^')
ax.scatter(xdata_ellipse[0], xdata_ellipse[1], k_ellipse, marker='o')

ax.set_zscale('log')
ax.view_init(elev=0, azim=90)  # Adjust the elevation and azimuthal angles as needed

ax.set_xlim([min(average_xdata_interp['Porosity']), max(average_xdata_interp['Porosity'])])
ax.set_ylim([min(average_xdata_interp['Surface']), max(average_xdata_interp['Surface'])])
ax.set_zlim([min(klist), max(klist)])

# Add labels and legend
ax.set_xlabel('Porosity')
ax.set_ylabel('Surface')
ax.set_zlabel('Permeability')
ax.legend()
plt.show()



#Plot the graphs individually with log scale
fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(12, 6))

# Plot Permeability vs Porosity
ax1.scatter(concatenated_df_rectangle['Porosity'], concatenated_df_rectangle['Permeability'], cmap='winter', marker='s')
ax1.scatter(concatenated_df_triangle['Porosity'], concatenated_df_triangle['Permeability'], cmap='winter', marker='^')
ax1.scatter(concatenated_df_ellipse['Porosity'], concatenated_df_ellipse['Permeability'], cmap='winter', marker='o')
ax1.set_xlabel('Porosity')
ax1.set_ylabel('Permeability')
#ax.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()


# Plot Permeability vs Surface

ax2.scatter(concatenated_df_rectangle['Surface'], concatenated_df_rectangle['Permeability'], cmap='winter', marker='s')
ax2.scatter(concatenated_df_triangle['Surface'], concatenated_df_triangle['Permeability'], cmap='winter', marker='^')
ax2.scatter(concatenated_df_ellipse['Surface'], concatenated_df_ellipse['Permeability'], cmap='winter', marker='o')
ax2.set_xlabel('Surface')
ax2.set_ylabel('Permeability')
#ax.set_xscale('log')
ax2.set_yscale('log')
ax2.legend()
plt.show()



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

'''