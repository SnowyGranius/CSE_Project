import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import perimeter, perimeter_crofton
from skimage.measure import euler_number
import os
import pandas as pd

for i in range (1, 11):
    for j in range(1, 11):
        for k in np.arange(0.1, 1.1, 0.1):
            # 1 = pore space, 0 = solid space
            im = ps.generators.blobs(shape = [500, 500], porosity = k, blobiness = j, seed = i)

            # Calculate MFs
            # M0
            M0 = ps.metrics.porosity(im)
            #print('M0 = {}'.format(M0))

            #M1
            mesh = ps.tools.mesh_region(region = im)
            M1 = ps.metrics.mesh_surface_area(mesh = mesh)
            #print('M1 = {}'.format(M1))

            per4 = perimeter(im, 4)
            per8 = perimeter(im, 8)
            per_c2 = perimeter_crofton(im, 2)
            per_c4 = perimeter_crofton(im, 4)

            # print('Perimeter, 4 connectivity = {}'.format(per4))
            # print('Perimeter, 8 connectivity = {}'.format(per8))
            # print('Crofton Perimeter, 2 connectivity = {}'.format(per_c2))
            # print('Crofton Perimeter, 4 connectivity = {}'.format(per_c4))

            im_inv = np.invert(im)
            M2 = euler_number(im_inv, connectivity = 1)

            #print('M2 = {}'.format(M2))\
            
            script_dir = os.path.dirname(__file__)
            base_path=script_dir
            sub_path='Microstructure_generation'

            path=os.path.join(base_path, sub_path)
        
            # Check if the file exists3
            new_data = pd.DataFrame([[i, j, M0, per4, M2]], columns=['Seed', 'Blobiness', 'M0', 'M1', 'M2'])

            csv_file = os.path.join(path, 'microstructure_data.csv')
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

            # Show microstructure
            # plt.imshow(im)
            # plt.colorbar()
            # plt.savefig(os.path.join(path, f'microstructure_seed_{i}_blobiness_{j}.png'))
            # plt.close()