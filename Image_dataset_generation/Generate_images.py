import numpy as np
from skimage.draw import disk
import matplotlib.pyplot as plt
import sys
import os

current_directory = os.path.dirname(os.path.abspath(sys.argv[0])) 
print(current_directory)

# Arrays to store circle data
x_coord = np.array([])
y_coord = np.array([])
radii = np.array([])


def read_circle_data(file_path):
    print(file_path)
    with open(file_path, 'r') as file:
        for line in file:
            parameters = line.strip().split()
            x, y, z, radius = map(float, parameters)

            # Append to numpy arrays
            global x_coord, y_coord, radii
            x_coord = np.append(x_coord, [x])
            y_coord = np.append(y_coord, [y])
            radii = np.append(radii, [radius])

def generate_image(image_name, image_shape=(1000, 1000)):
    # Create a blank binary image
    image = np.zeros(image_shape, dtype=np.uint8)

    # Create scalers to scale the coordinates to the image size
    x_min = np.min(x_coord)
    x_max = np.max(x_coord)
    y_min = np.min(y_coord)
    y_max = np.max(y_coord)
    x_scaler = (image_shape[1] - 1) / (x_max - x_min)
    y_scaler = (image_shape[0] - 1) / (y_max - y_min)
    print(x_scaler, y_scaler)

    
    x_coord_scaled = (x_coord - x_min) * x_scaler
    y_coord_scaled = (y_coord - y_min) * y_scaler
    radii_scaled = radii * x_scaler # Or y scaler?

    
    nr_circles = len(x_coord)
    for i in range(nr_circles):
        rr, cc = disk((x_coord_scaled[i], y_coord_scaled[i]), radii_scaled[i], shape=image_shape)
        image[rr, cc] = 1 # Foreground

    
    plt.imsave(f'{current_directory}/Circle_Images/{image_name}', image, cmap='gray')

Models = [1]
pfs = ['0.1']
# Models = range(1, 26)
# pfs = np.arange(1, 11, 1) / 10
# print(pfs)

for model in Models:
    for pf in pfs:
        input_file = f'Model_{model}_pf_{pf}00.txt'
        circles = read_circle_data(f'{current_directory}/Circle_data_porespy/{input_file}')
        input_file = input_file.split('.t')[0]
        generate_image(image_name=f'{input_file}.png')
        # print("New Image Generated")
        # Empty the arrays for the next iteration
        x_coord = np.array([])
        y_coord = np.array([])
        radii = np.array([])
