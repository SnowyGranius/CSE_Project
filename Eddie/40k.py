import porespy as ps
import numpy as np
from skimage.measure import perimeter, perimeter_crofton
from skimage.measure import euler_number
import os
import pandas as pd

#b=10
#p=0.55
resolution=800
im_nrs=20

script_dir = os.path.dirname(__file__)
base_path=script_dir
sub_path='DAT_files'
path=os.path.join(base_path, sub_path)
#path=os.path.join(base_path, sub_path, f'blobiness_{b}_porosity_{p}')


for p in np.linspace(0.55, 0.95, 10):
    for b in np.linspace(10, 14, 5):
        new_folder = f'blobiness_{b}_porosity_{p}'
        new_folder_path = os.path.join(path, new_folder)
        os.makedirs(new_folder_path, exist_ok=True)
        # Generate the original image
        im = ps.generators.blobs(shape=[resolution, resolution], porosity=p, blobiness=b, seed=np.random.randint(0, 40000))
        im_inv = np.invert(im)

        #-------------------Calculate the M0, M1, and M2 for the original image-------------------
        #M0_original = ps.metrics.porosity(im)
        #M1_original = perimeter_crofton(im, 4)
        #M2_original = euler_number(im_inv, connectivity = 1)


        # Split the original image into subimages
        #subimages = []
        subimage_size = resolution // (im_nrs)  # Define the size of each subimage
        M0_list = []
        M1_list = []
        M2_list = []
        avg_data_list = []

        for i in range(0, resolution, subimage_size):
            for j in range(0, resolution, subimage_size):
                sub=im[i:i + subimage_size, j:j + subimage_size]
                sub_file_path = os.path.join(path, new_folder_path, f'subimage_{i}_{j}.dat')
                #subimages.append(sub)
        # Calculate the M0, M1, and M2 for each subimage
                M0=ps.metrics.porosity(sub)
                M1=perimeter_crofton(sub, 4)
                M2=euler_number(np.invert(sub), connectivity = 1)
                M0_list.append(M0)
                M1_list.append(M1)
                M2_list.append(M2)

                with open(sub_file_path, 'a') as f:
                    np.savetxt(f, sub, fmt='%s', delimiter=' ')
                    np.savetxt(f, [M0, M1, M2], fmt='%s', delimiter=' ')
        # Calculate the average M0, M1, and M2 for the subimages
        M0_avg = np.mean(M0_list)
        M1_avg = np.mean(M1_list)
        M2_avg = np.mean(M2_list)

        avg_data_list.append([M0_avg, M1_avg, M2_avg])
        #output_file=f'porespy_data_{b}_{p}_{resolution}_{im_nrs}.dat'
        for i in range(0, resolution, subimage_size):
            for j in range(0, resolution, subimage_size):
                sub_file_path = os.path.join(path, new_folder_path, f'subimage_{i}_{j}.dat')
                with open(sub_file_path, 'a') as f:
                    np.savetxt(f, avg_data_list, fmt='%s', delimiter=' ')

        #print(subimages)
print('Done')
