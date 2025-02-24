## synthetic_microstructure_test.py


This script is used to test the generation and analysis of synthetic microstructures. It performs the following tasks:
- Defines theoretical values for Minkowski functionals (M0, M1, and M2) for different shapes - taken from external simulations.
- Initializes arrays to store calculated values of M0, M1, and M2 for different shapes and resolutions - using skimage and porespy functions.
- Iterates over a range of resolutions and shapes to generate synthetic microstructure images.
- Loads the generated images from specified file paths.
- Calculates the Minkowski functionals (M0, M1, and M2) for each image.
- Chooses the perimeter value closest to the theoretical value for M1.
- Returns the calculated values of M0, M1, and M2 for further analysis.

The function created in this file is used in 'plot_error_mf_computation_vs_synthetic.py'.

## plot_error_mf_computation_vs_synthetic.py

This script is used to plot the errors between the measured and theoretical Minkowski functionals (M0, M1, and M2) for synthetic microstructures. It performs the following tasks:
- Imports the calculated values of M0, M1, and M2 from 'synthetic_microstructure_test.py'.
- Defines theoretical values for Minkowski functionals (M0, M1, and M2) for different shapes.
- Calculates the relative errors between the measured and theoretical values for M2.
- Plots the relative errors for M2 as a function of resolution for different shapes.
- Adds labels, titles, and legends to the plot for better visualization.
- Displays the plot.