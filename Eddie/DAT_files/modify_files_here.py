import os
import csv
import pandas as pd

# Define the paths
base_dir = os.path.dirname(__file__)
subfolder = os.path.join(base_dir, 'blobiness_10.0_porosity_0.65_2910')  # Replace 'subfolder_name' with the actual subfolder name
csv_file_path = os.path.join(subfolder, 'Minkowskis_blob_10_pf_0.640_porespy.csv')  # Replace 'your_file.csv' with the actual CSV file name

# Read the CSV file
csv_data = pd.read_csv(csv_file_path)
print(csv_data)
'''
# Iterate through the rows of the CSV file
for index, row in csv_data.iterrows():
    # Get the numbers from the first two columns
    num1 = row[0]
    num2 = row[1]
    
    # Construct the .dat file name based on the numbers
    dat_file_name = f"subimage_{int(num1)}_{int(num2)}.dat"
    dat_file_path = os.path.join(subfolder, dat_file_name)
    
    # Read the .dat file
    with open(dat_file_path, 'r') as dat_file:
        dat_lines = dat_file.readlines()
        
        # Check if the file has at least 83 lines
        if len(dat_lines) >= 83:
            line_83 = dat_lines[82].strip()  # Line 83 is at index 82
        else:
            line_83 = None  # Handle the case where the file has less than 83 lines
    
    # Add the extracted line 83 to a new column in the CSV data
    csv_data.at[index, 'Euler_Characteristic'] = line_83

# Write the updated data back to the CSV file
csv_data.to_csv(csv_file_path, index=False)

print("Data from line 83 of .dat files has been added to the CSV file.")
'''
# Divide the values in the "Porosity" column by 100
csv_data['Porosity'] = csv_data['Porosity'] / 100

# Write the updated data back to the CSV file
csv_data.to_csv(csv_file_path, index=False)

print("Data from line 83 of .dat files has been added to the CSV file and 'Porosity' values have been divided by 100.")