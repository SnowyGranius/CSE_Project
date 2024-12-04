import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import perimeter, perimeter_crofton
from skimage.measure import euler_number
import os
import pandas as pd

def read_dat_file(file_path):
    with open(file_path, 'r') as file:
        lines = [next(file) for _ in range(40)]
    return lines

def reconstruct_image(lines):
    data = [list(map(lambda x: 1 if x == 'True' else 0, line.split())) for line in lines]
    image = np.array(data, dtype=np.uint8)
    return image

script_dir = os.path.dirname(__file__) # <-- absolute dir the script is in
base_path=script_dir
sub_path='DAT_files'
path=os.path.join(base_path, sub_path, 'blobiness_10_porosity_0.55')
file_to_read='subimage_0_0.dat'
file_path=os.path.join(path, file_to_read)
lines = read_dat_file(file_path)
image = reconstruct_image(lines)

plt.imshow(image, cmap='gray')
plt.show()