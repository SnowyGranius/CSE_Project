import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

# A program that reads data from a csv file and compares features using a pairplot

# # Read the data from the csv file
# df = pd.read_csv('C:\\Users\\ionst\\Documents\\Fisiere Python\\Proiect\\Data\Datasets\\Heterogenous_samples\\Minkowskis_pf_0.220_rectangle_1_hetero.csv')
# print(df)

# Read all relevant files in the folder

path = 'C:\\Users\\ionst\\Documents\\Fisiere_Python\\Proiect\\Data\\Datasets\\Heterogenous_samples\\'
# all_files = glob.glob(os.path.join(path, "*0.300*.csv"))
all_files = glob.glob(os.path.join(path, "*rectangle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_rectangle = pd.concat(df_from_each_file, ignore_index=True)
# multiply the column permeability with 1e5 to get the values in the same range
concatenated_df_rectangle['Permeability'] = concatenated_df_rectangle['Permeability'] * 1e5
concatenated_df_rectangle['Energy'] = concatenated_df_rectangle['Energy'] * 1e5


concatenated_df_rectangle['Euler_total'] = 'Rectangle'

sns.pairplot(concatenated_df_rectangle, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], hue='Permeability', palette='viridis', markers='s')
sns.pairplot(concatenated_df_rectangle, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], markers='s', diag_kind='kde')

all_files = glob.glob(os.path.join(path, "*triangle*.csv"))
df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
concatenated_df_triangle = pd.concat(df_from_each_file, ignore_index=True)
print(concatenated_df_triangle)
concatenated_df_triangle['Euler_total'] = 'Triangle'
concatenated_df_triangle['Permeability'] = concatenated_df_triangle['Permeability'] * 1e5
concatenated_df_triangle['Energy'] = concatenated_df_triangle['Energy'] * 1e5

sns.pairplot(concatenated_df_triangle, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], hue='Permeability', palette='viridis', markers='^')
sns.pairplot(concatenated_df_triangle, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], markers='^', diag_kind='kde')

# all_files = glob.glob(os.path.join(path, "*ellipse*.csv"))
# df_from_each_file = (pd.read_csv(f).mean(axis=0).to_frame().T for f in all_files)
# concatenated_df_ellipse = pd.concat(df_from_each_file, ignore_index=True)
# concatenated_df_ellipse['Euler_total'] = 'Ellipse'
# concatenated_df_ellipse['Permeability'] = concatenated_df_ellipse['Permeability'] * 1e5
# concatenated_df_ellipse['Energy'] = concatenated_df_ellipse['Energy'] * 1e5


# sns.pairplot(concatenated_df_ellipse, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], hue='Permeability', palette='viridis', markers='o')
# sns.pairplot(concatenated_df_ellipse, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], markers='o', diag_kind='kde')

concated_df = pd.concat([concatenated_df_rectangle, concatenated_df_triangle], ignore_index=True)
print(concated_df)

#sns.pairplot(concated_df, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], hue='Euler_total')
sns.pairplot(concated_df, vars=['Porosity','Surface','Euler_mean_vol','Permeability','Energy'], markers=['s','^'])

# save the dataframe concated_df to a csv file
# concated_df.to_csv('C:\\Users\\ionst\\Documents\\Fisiere_Python\\Proiect\\Data\\Datasets\\Threshold_homogenous_diamater_wide_RCP\\concatenated_df.csv', index=False)

# Display the plot
plt.show()
