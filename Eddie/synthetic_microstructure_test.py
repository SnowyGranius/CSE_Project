import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import perimeter, perimeter_crofton
from skimage.measure import euler_number
import os
#import pandas as pd
def array_creation():
    shapes = ["circle", "rectangle", "ellipse", "triangle"]
    M0_theoretical_values = [0.8876751204372201, 0.92831114591458, 0.9436870855472399, 0.95670243022806]

    M1_theoretical_values = [44.773683110127, 40.362048941883, 34.654736584218995, 36.561243576213]

    M2_theoretical_values = [1442, 1442, 1442, 1442]

    M0_values = np.zeros((len(shapes), 10))
    M1_values = np.zeros((len(shapes), 10))
    M2_values = np.zeros((len(shapes), 10))

    for i in range(1000, 11000, 1000):
        for k in range(len(shapes)):
            # 1 = pore space, 0 = solid space
            # np.random.seed(10)
            nr_images = 20

            #im = ps.generators.blobs(shape = [i, i], porosity = 0.55, blobiness = 10, seed=14341)
            script_dir = os.path.dirname(__file__)
            image_path = os.path.join(script_dir, 'Synthetic_Image_Analysis', 'Full_Images_1000-10000', f'pf_0.100_{shapes[k]}_Model_2_resolution_({i}, {i}).png')
            im = plt.imread(image_path)
            im = im[:, :, 0]
            im = np.invert(im.astype(bool))
            #plt.imshow(im, interpolation='none', cmap='gray')
            #plt.show()


            #profile=ps.metrics.representative_elementary_volume(im=im)
            # Calculate MFs
            # M0
            M0 = ps.metrics.porosity(im)
            #print('M0 = {}'.format(M0))

            #M1
            #mesh = ps.tools.mesh_region(region = im)
            #M1 = ps.metrics.mesh_surface_area(mesh = mesh)
            #print('M1 = {}'.format(M1))

            #per4 = perimeter(im, 4)/i
            #per8 = perimeter(im, 8)/i
            per_c2 = perimeter_crofton(im, 2)/i
            per_c4 = perimeter_crofton(im, 4)/i

            # Choose the perimeter closest to the theoretical value
            theoretical_value = M1_theoretical_values[k]
            if abs(per_c2 - theoretical_value) < abs(per_c4 - theoretical_value):
                chosen_perimeter = per_c2
            else:
                chosen_perimeter = per_c4

            #print('Chosen Perimeter = {}'.format(chosen_perimeter))

            #print('Perimeter, 4 connectivity = {}'.format(per4))
            #print('Perimeter, 8 connectivity = {}'.format(per8))
            #print('Crofton Perimeter, 2 connectivity = {}'.format(per_c2))
            #print('Crofton Perimeter, 4 connectivity = {}'.format(per_c4))

            im_inv = np.invert(im)
            M2 = euler_number(im_inv, connectivity = 1)



            #print('M2 = {}'.format(M2))

            M0_values[k, (i // 1000) - 1] = M0
            M1_values[k, (i // 1000) - 1] = chosen_perimeter
            M2_values[k, (i // 1000) - 1] = M2
    return M0_values, M1_values, M2_values

'''
script_dir = os.path.dirname(__file__)
base_path=script_dir
sub_path='Microstructure_generation'

path=os.path.join(base_path, sub_path)


# Check if the file exists
new_data = pd.DataFrame([[10, 2, M0, per4, M2]], columns=['Seed', 'Blobiness', 'M0', 'M1', 'M2'])

csv_file = os.path.join(path, 'shape_microstructure_var.csv')
file_exists = os.path.isfile(csv_file)

if file_exists:
    # If the file exists, read it into a DataFrame
    df = pd.read_csv(csv_file)
    # Append the new data to the DataFrame
    df = pd.concat([df, new_data], ignore_index=True)
else:
    # If the file does not exist, use the new data as the DataFrame
    df = new_data

# Write the DataFrame back to the CSV file
df.to_csv(csv_file, index=False)
'''
'''
# Show microstructure
# plt.imshow(im)
# plt.colorbar()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Black and white image
ax[0].imshow(im, cmap='gray', interpolation='none')
ax[0].set_title('Black and White Microstructure')
ax[0].axis('off')

# Color image
ax[1].imshow(im, interpolation='none')
ax[1].set_title('Color Microstructure')
ax[1].axis('off')

#plt.colorbar(ax[1].images[0], ax=ax[1])
#plt.show()
#plt.savefig(os.path.join(path, f'resolution_{i}_seed_10_blobiness_2.png'))
#plt.close()
'''