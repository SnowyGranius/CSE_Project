import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import perimeter, perimeter_crofton
from skimage.measure import euler_number
import csv
import os

for i in range (1, 51):
    for j in range(1, 11):
        # 1 = pore space, 0 = solid space
        im = ps.generators.blobs(shape = [500, 500], porosity = 0.6, blobiness = j, seed = i)

        # Calculate MFs
        # M0
        M0 = ps.metrics.porosity(im)
        print('M0 = {}'.format(M0))

        #M1
        mesh = ps.tools.mesh_region(region = im)
        M1 = ps.metrics.mesh_surface_area(mesh = mesh)
        print('M1 = {}'.format(M1))

        per4 = perimeter(im, 4)
        per8 = perimeter(im, 8)
        per_c2 = perimeter_crofton(im, 2)
        per_c4 = perimeter_crofton(im, 4)

        print('Perimeter, 4 connectivity = {}'.format(per4))
        print('Perimeter, 8 connectivity = {}'.format(per8))
        print('Crofton Perimeter, 2 connectivity = {}'.format(per_c2))
        print('Crofton Perimeter, 4 connectivity = {}'.format(per_c4))

        im_inv = np.invert(im)
        M2 = euler_number(im, connectivity = 1)

        print('M2 = {}'.format(M2))\
        
        script_dir = os.path.dirname(__file__)
        base_path=script_dir
        path='Microstructure generation'

        path=os.path.join(base_path, path)
    
        # Check if the file exists3
        file_exists = os.path.isfile(path)

        # Open the CSV file in append mode
        with open(path, newline='') as file:
            writer = csv.writer(file)
            # Write the header only if the file does not exist
            if not file_exists:
                writer.writerow(['Seed', 'Blobiness', 'M0', 'M1', 'M2'])
                # Write the data
                writer.writerow([i, j, M0, M1, M2])
        # Show microstructure
        plt.imshow(im)
        plt.colorbar()
        plt.savefig(os.path.join(path, f'microstructure_seed_{i}_blobiness_{j}.png'))
        plt.close()