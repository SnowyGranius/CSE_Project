# MICRO-GENERATION Project

This project involves generating and analyzing microstructures using various Python scripts. Below is a brief explanation of each script in the project:

## Plotting_microstructure.py

This script is used to plot the Minkowski space to find the relationship between different microstructure parameters (M0, M1, and M2) and the blobiness of the microstructures. It performs the following tasks:
- Loads microstructure data from a CSV file.
- Filters the data based on specific criteria.
- Creates a 3D scatter plot of the filtered data, with blobiness values represented by color.
- Animates the 3D plot by rotating it.
- Fits a surface to the data points and plots the fitted surface along with the original data points.

## Image_generator.py

This script generates microstructure images and calculates various metrics for each image. It performs the following tasks:
- Generates random microstructure images using the `porespy` library.
- Splits the generated images into subimages.
- Calculates the Minkowski functionals (M0, M1, and M2) for each subimage.
- Saves the subimages and their corresponding metrics to DAT files.
- Calculates the average Minkowski functionals for the subimages and saves them to the DAT files.

## Image_checker.py

This script reads and reconstructs microstructure images from DAT files and saves them as PNG images. It performs the following tasks:
- Reads the DAT files containing the subimages and their metrics.
- Reconstructs the subimages from the DAT files.
- Saves each subimage as a PNG file.
- Combines the subimages into a larger image and saves it as a PNG file.



