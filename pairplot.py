import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from ellipse_marker import ellipse
# A program that reads data from a csv file and compares features using a pairplot

# # Read the data from the csv file
# df = pd.read_csv('C:\\Users\\ionst\\Documents\\Fisiere Python\\Proiect\\Data\Datasets\\Heterogenous_samples\\Minkowskis_pf_0.220_rectangle_1_hetero.csv')
# print(df)

# Read all relevant files in the folder
script_dir = os.path.dirname(__file__)
sub_path = 'Porespy_homogenous_diamater'
path = os.path.join(script_dir, sub_path)
# all_files = glob.glob(os.path.join(path, "*0.300*.csv"))
all_files = glob.glob(os.path.join(path, "*rectangle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_rectangle = pd.concat(df_from_each_file, ignore_index=True)
concatenated_df_rectangle.rename(columns={'Euler_total':'Shape'}, inplace=True)
concatenated_df_rectangle['Shape'] = 'Rectangle'
#change the name of concatenated_df_rectangle['Euler_mean_vol'] to Euler Characteristic
concatenated_df_rectangle.rename(columns={'Euler_mean_vol':'Euler Characteristic'}, inplace=True)
concatenated_df_rectangle['Permeability'] = concatenated_df_rectangle['Permeability']
concatenated_df_rectangle['Energy'] = concatenated_df_rectangle['Energy']



# sns.pairplot(concatenated_df_rectangle, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], hue='Permeability', palette='viridis', markers='s')
# sns.pairplot(concatenated_df_rectangle, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], markers='s', diag_kind='kde')

all_files = glob.glob(os.path.join(path, "*triangle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_triangle = pd.concat(df_from_each_file, ignore_index=True)
concatenated_df_triangle.rename(columns={'Euler_total':'Shape'}, inplace=True)
# print(concatenated_df_triangle)
concatenated_df_triangle['Shape'] = 'Triangle'
concatenated_df_triangle.rename(columns={'Euler_mean_vol':'Euler Characteristic'}, inplace=True)
concatenated_df_triangle['Permeability'] = concatenated_df_triangle['Permeability']
concatenated_df_triangle['Energy'] = concatenated_df_triangle['Energy']

# sns.pairplot(concatenated_df_triangle, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], hue='Permeability', palette='viridis', markers='^')
# sns.pairplot(concatenated_df_triangle, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], markers='^', diag_kind='kde')

all_files = glob.glob(os.path.join(path, "*ellipse*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_ellipse = pd.concat(df_from_each_file, ignore_index=True)
concatenated_df_ellipse.rename(columns={'Euler_total':'Shape'}, inplace=True)
concatenated_df_ellipse['Shape'] = 'Ellipse'
concatenated_df_ellipse.rename(columns={'Euler_mean_vol':'Euler Characteristic'}, inplace=True)
concatenated_df_ellipse['Permeability'] = concatenated_df_ellipse['Permeability']
concatenated_df_ellipse['Energy'] = concatenated_df_ellipse['Energy']


# sns.pairplot(concatenated_df_ellipse, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], hue='Permeability', palette='viridis', markers='o')
# sns.pairplot(concatenated_df_ellipse, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], markers=ellipse, diag_kind='kde')

all_files = glob.glob(os.path.join(path, "*circle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_circle = pd.concat(df_from_each_file, ignore_index=True)
concatenated_df_circle.rename(columns={'Euler_total':'Shape'}, inplace=True)
concatenated_df_circle['Shape'] = 'Circle'
concatenated_df_circle.rename(columns={'Euler_mean_vol':'Euler Characteristic'}, inplace=True)
concatenated_df_circle['Permeability'] = concatenated_df_circle['Permeability']
concatenated_df_circle['Energy'] = concatenated_df_circle['Energy']


# sns.pairplot(concatenated_df_circle, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], hue='Permeability', palette='viridis', markers='o')
# sns.pairplot(concatenated_df_circle, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], markers='o', diag_kind='kde')

concated_df = pd.concat([concatenated_df_rectangle, concatenated_df_triangle, concatenated_df_ellipse, concatenated_df_circle], ignore_index=True)
print(concated_df)

#sns.pairplot(concated_df, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], hue='Euler_total')
# sns.pairplot(concated_df, vars=['Porosity','Surface','Euler Characteristic','Permeability','Energy'], markers=['s','^', ellipse, 'o'], hue='Shape', diag_kind='kde')

# save the dataframe concated_df to a csv file
# concated_df.to_csv('C:\\Users\\ionst\\Documents\\Fisiere_Python\\Proiect\\Data\\Datasets\\Threshold_homogenous_diamater_wide_RCP\\concatenated_df.csv', index=False)

# Display the plot
# save the plot in the folder called Porespy_plots
# plt.savefig(os.path.join(script_dir, 'Porespy_plots', 'initial_analysis.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)

#plot surface vs permeability
sns.scatterplot(data=concated_df, x='Porosity', y='Permeability', hue='Shape', style='Shape', markers=['s','^', ellipse, 'o'])
plt.xlabel('Porosity')
plt.ylabel('Permeability')
plt.title('Porosity vs Permeability')
plt.grid()
plt.savefig(os.path.join(script_dir, 'Porespy_plots', 'Porosity_vs_Permeability.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()