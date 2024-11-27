from PIL import Image, ImageDraw
import sys
import os

current_directory = os.path.dirname(os.path.abspath(sys.argv[0])) 
print(current_directory)


def read_circle_data(file_path):
    print(file_path)
    circles = []
    with open(file_path, 'r') as file:
        counter = 0
        for line in file:
            parameters = line.strip().split()
            x, y, z, radius = map(float, parameters)
            if (counter % 10 == 0):
                print(x, y, z, radius)
            circles.append((x, y, radius))
            counter += 1
    return circles

def generate_image(circles, image_size=(1000, 1000), output_path=f'{current_directory}/Circle_Images/output_image.png'):
    
    for circle in circles:
        
        x, y, radius = circle
        
        # Scaling coordinates and radius to fit the image size (if necessary)
        x = x % image_size[0]
        y = y % image_size[1] 
        radius = max(1, min(radius, min(image_size) // 2))  
        


input_file = 'Model_1_pf_0.070_rad_0.005_centers.txt'  # Replace with your input file path
circles = read_circle_data(f'{current_directory}/Circle_data/{input_file}')
generate_image(circles)