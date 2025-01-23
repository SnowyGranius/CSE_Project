import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import perimeter, perimeter_crofton
from skimage.measure import euler_number
import os
import pandas as pd
resolution_spacing = 80

def read_dat_file(file_path):
    with open(file_path, 'r') as file:
        lines = [next(file) for _ in range(resolution_spacing)]
    return lines

def reconstruct_image(lines):
    data = [list(map(lambda x: 1 if x == 'True' else 0, line.split())) for line in lines]
    image = np.array(data, dtype=np.uint8)
    return image

script_dir = os.path.dirname(__file__) # <-- absolute dir the script is in
base_path=script_dir
sub_path='DAT_files'
folder='blobiness_10.0_porosity_0.65_2910'

path=os.path.join(base_path, sub_path, folder)
file_to_read='subimage_0_0.dat'
file_path=os.path.join(path, file_to_read)
lines = read_dat_file(file_path)
sub_image = reconstruct_image(lines)

another_folder='images'
save_img_path=os.path.join(path, another_folder, 'sub_image.png')

plt.imshow(sub_image, cmap='gray')
#plt.title(f'{folder}, {file_to_read}')
plt.show()



sub_images = []
for i in range(0, 800, resolution_spacing):
    for j in range(0, 800, resolution_spacing):
        file_name = f'subimage_{i}_{j}.dat'
        file_path = os.path.join(path, file_name)
        if os.path.exists(file_path):
            #print(file_name)
            lines = read_dat_file(file_path)
            sub_image = reconstruct_image(lines)
            sub_images.append(sub_image)

            # Save each sub-image as a PNG file
            save_img_path = os.path.join(path, another_folder)
            os.makedirs(save_img_path, exist_ok=True)
            file_name = f'subimage_{i}_{j}.png'
            save_path = os.path.join(save_img_path, file_name)
            plt.imsave(save_path, sub_image, cmap='gray')

# Assuming all sub-images are of the same size
rows = cols = int(np.sqrt(len(sub_images)))
sub_image_shape = sub_images[0].shape
print(sub_image_shape)
large_image = np.zeros((rows * sub_image_shape[0], cols * sub_image_shape[1]), dtype=np.uint8)
# print(sub_image_shape)
# print(large_image.shape)
# print(rows, cols)

for i in range(rows):
    for j in range(cols):
        large_image[i * sub_image_shape[0]:(i + 1) * sub_image_shape[0], j * sub_image_shape[1]:(j + 1) * sub_image_shape[1]] = sub_images[i * cols + j]

large_image_path=os.path.join(save_img_path, 'large_image.png')
plt.imsave(large_image_path, large_image, cmap='gray')
#plt.title(f'{folder}')
plt.show()

# Save the sub_image
# sub_image_save_path = os.path.join(base_path, 'sub_image.png')
# plt.imsave(sub_image_save_path, sub_image, cmap='gray')




