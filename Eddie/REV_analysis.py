import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import perimeter, perimeter_crofton
from skimage.measure import euler_number
import os
import pandas as pd

DATA_GEN=True
DATA_ANALYSIS=True
DATA_PLOTTING=True
make_subfolder=False
image_generation=False

#Keep blobiness fixed
b=15
p=0.95

#subsample nr per side
im_nrs=20

#------------------IMAGE & DATA GENERATION---------------#
if DATA_GEN:
    for resolution in range(50, 650, 50):
        #resolution=800
        #os.makedirs(f'subimages_{resolution}_porosity_{p}_blob_{b}', exist_ok=True)
        # 1 = pore space, 0 = solid space
        grid_size = 25  # Define the size of the grid
        #fig1, ax1 = plt.subplots(1, 1, figsize=[6, 6])
        #fig2, ax2 = plt.subplots(1, 1, figsize=[6, 6])

        im = ps.generators.blobs(shape=[resolution, resolution], porosity=p, blobiness=b, seed=10)
        # Create a mesh grid for subsampling
        x = np.linspace(0, resolution, grid_size)
        y = np.linspace(0, resolution, grid_size)
        xv, yv = np.meshgrid(x, y)

        # Subsample the image using the mesh grid
        # Ensure indices are within bounds
        xv = np.clip(xv, 0, resolution - 1)
        yv = np.clip(yv, 0, resolution - 1)
        subsampled_im = im[yv.astype(int), xv.astype(int)]

        # Plot the original image
        # ax1.imshow(im, cmap='gray')
        # ax1.set_title('Original Image')
        # ax1.axis('off')
        # plt.show()


        # Plot the subsampled image
        script_dir = os.path.dirname(__file__)
        base_path=script_dir
        sub_path='REV_files'
        path=os.path.join(base_path, sub_path, 'Resolution_Determination')
        if make_subfolder:
            new_folder = f'subimages_{resolution}_porosity_{p}_blob_{b}'
            new_folder_path = os.path.join(path, new_folder)
            os.makedirs(new_folder_path, exist_ok=True)
        for k in range(im_nrs, im_nrs+1):
            if resolution % k==0:
                # Split the original image into subimages
                subimages = []
                subimage_size = resolution // (k)  # Define the size of each subimage
                for i in range(0, resolution, subimage_size):
                    for j in range(0, resolution, subimage_size):
                        sub=im[i:i + subimage_size, j:j + subimage_size]
                        subimages.append(sub)

                        M0 = ps.metrics.porosity(sub)
                        per4 = perimeter_crofton(sub, 4)
                        sub_inv = np.invert(sub)
                        M2 = euler_number(sub_inv, connectivity = 1)
                        # Check if the file exists
                        new_data = pd.DataFrame([[resolution, k**2, p, b, M0, per4, M2]], columns=['Resolution','Subsamples', 'Porosity', 'Blobiness', 'M0', 'M1', 'M2'])

                        csv_file = os.path.join(path, f'porosity_{p}_individual_MF_blobiness_{b}_sub_{im_nrs}.csv')
                        file_exists = os.path.isfile(csv_file)

                        if file_exists:
                            # If the file exists, read it into a DataFrame
                            df = pd.read_csv(csv_file)
                            # # Append the new data to the DataFrame
                            df = pd.concat([df, new_data], ignore_index=True)
                        else:
                            # If the file does not exist, use the new data as the DataFrame
                            df = new_data
                        # Write the DataFrame back to the CSV file
                        df.to_csv(csv_file, index=False)
                if image_generation:
                    # Plot all subimages in one figure
                    fig, axes = plt.subplots(k, k, figsize=[12, 12])
                    # Ensure axes is always a 2D array, even if k=1
                    for idx, subimage in enumerate(subimages):
                        if idx < k * k:
                            ax = axes[idx // k, idx % k]
                            ax.imshow(subimage, cmap='gray', interpolation='none')
                            ax.axis('off')
                            ax.set_title(f'Subimage {idx + 1}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(new_folder_path, f'subimages_{resolution}_{k}.png'))
                    #plt.close()
                    
                else:
                    pass


#------------------DATA PLOTTING-------------------------#
if DATA_ANALYSIS:
    script_dir = os.path.dirname(__file__)
    base_path=script_dir
    path=os.path.join(base_path, 'REV_files', 'Resolution_Determination', f'porosity_{p}_individual_MF_blobiness_{b}_sub_{im_nrs}.csv')
    data = pd.read_csv(path)

    # Extract M0, M1, and M2 columns
    M0 = data['M0']
    M1 = data['M1']/4
    M2 = data['M2']
    subsamples=data['Subsamples']
    ###--------------DATA FILTERING---------------###
    if DATA_PLOTTING:
        resolutions = data['Resolution'].unique()
        fig, axes = plt.subplots(1, len(resolutions), figsize=(15, 5), sharey=True)
        axes = np.atleast_1d(axes)
        for idx, resolution in enumerate(resolutions):
            ax = axes[idx]
            ax.set_title(f'Resolution {resolution}')
            filtered_data = data[data['Resolution'] == resolution]
            subsample_nr = filtered_data['Subsamples']
            
            M0_filtered = filtered_data[filtered_data['Subsamples'] == 100]['M0']
            M1_filtered = filtered_data[filtered_data['Subsamples'] == 100]['M1']
            M2_filtered = filtered_data[filtered_data['Subsamples'] == 100]['M2']
            avg_M0 = filtered_data[filtered_data['Subsamples'] == 100].groupby(1/subsample_nr)['M0'].mean()
            avg_M1 = filtered_data[filtered_data['Subsamples'] == 100].groupby(1/subsample_nr)['M1'].mean()
            avg_M2 = filtered_data[filtered_data['Subsamples'] == 100].groupby(1/subsample_nr)['M2'].mean()
            #print(subsample_nr, avg_M1)
            M0_std = filtered_data[filtered_data['Subsamples'] == 100].groupby(1/subsample_nr)['M0'].std()
            M1_std = filtered_data[filtered_data['Subsamples'] == 100].groupby(1/subsample_nr)['M1'].std()
            M2_std = filtered_data[filtered_data['Subsamples'] == 100].groupby(1/subsample_nr)['M2'].std()
            #print("Rel. Error", M0_std/(np.sqrt(subsample_nr.iloc[0])*avg_M0), M1_std/(np.sqrt(subsample_nr.iloc[0])*avg_M1), M2_std/(np.sqrt(subsample_nr.iloc[0])*avg_M2))
            ax.scatter(avg_M0.index, avg_M0.values, color='red', label='Average M0') 
            #scatter = ax.scatter(1/subsample_nr, M0_filtered, label=f'Resolution {resolution}')
            ax.set_xlabel('1/Subsamples')
            ax.set_ylabel('M0') 
        plt.tight_layout()
        plt.show()
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        # Plot M0
        ax = axes[0]
        ax.set_title(f'Mean M0 vs Resolution, blobiness={b}, porosity={p}')
        for resolution in resolutions:
            filtered_data = data[data['Resolution'] == resolution]
            avg_M0 = filtered_data.groupby('Resolution')['M0'].mean()
            ax.plot(avg_M0.index/np.sqrt((filtered_data['Subsamples'].iloc[0])), avg_M0.values, marker='o', label=f'Resolution {resolution}')
        ax.set_ylabel('Mean M0')
        #ax.legend()
        
        # Plot M1
        ax = axes[1]
        ax.set_title('Mean M1 vs Resolution')
        for resolution in resolutions:
            filtered_data = data[data['Resolution'] == resolution]
            avg_M1 = filtered_data.groupby('Resolution')['M1'].mean()
            subimage_resolution = resolution // int(np.sqrt(filtered_data['Subsamples'].iloc[0]))
            ax.plot(avg_M1.index/np.sqrt((filtered_data['Subsamples'].iloc[0])), avg_M1.values/subimage_resolution, marker='o', label=f'Resolution {resolution}')
        ax.set_ylabel('Mean M1')
        #ax.legend()

        # Plot M2
        ax = axes[2]
        ax.set_title('Mean M2 vs Resolution')
        for resolution in resolutions:
            filtered_data = data[data['Resolution'] == resolution]
            avg_M2 = filtered_data.groupby('Resolution')['M2'].mean()
            ax.plot(avg_M2.index/np.sqrt((filtered_data['Subsamples'].iloc[0])), avg_M2.values, marker='o', label=f'Resolution {resolution}')
        ax.set_xlabel('Resolution')
        ax.set_ylabel('Mean M2')
        #ax.legend()

        plt.tight_layout()
        plt.show()


# For 10x10, determine the desired resolution.



    # ax.legend()
    # # Extract M0, M1, and M2 columns for the filtered data
    # M0_filtered = data['M0']
    # print(M0_filtered)
    # subsample_nr=data['Subsamples']

    # # Plot the filtered data
    # scatter = ax.scatter(1/subsample_nr, M0_filtered)
    # ax.set_xlabel('1/Subsamples')
    # ax.set_ylabel('M0')
    # plt.show()



###------------------PLOT GENERATION--------------------###
'''
def generate_plots_with_averages(start, stop, step, resolution=200):
    fig, axes = plt.subplots(1, (stop - start) // step, figsize=[6 * ((stop - start) // step), 6])
    for idx, i in enumerate(range(start, stop, step)):
        # Generate the blob structure
        im = ps.generators.blobs(shape=[resolution, resolution], porosity=0.55, blobiness=i, seed=10)
        # Create a mesh grid for subsampling
        grid_size = 25  # Define the size of the grid
        x = np.linspace(0, resolution, grid_size)
        y = np.linspace(0, resolution, grid_size)
        xv, yv = np.meshgrid(x, y)
        
        # Subsample the image using the mesh grid
        # Ensure indices are within bounds
        xv = np.clip(xv, 0, resolution - 1)
        yv = np.clip(yv, 0, resolution - 1)
        subsampled_im = im[yv.astype(int), xv.astype(int)]

        # Plot the subsampled image
        axes[idx].imshow(subsampled_im, cmap='gray')
        axes[idx].set_title(f'Subsampled Image for i={i}')
        axes[idx].axis('off')
        
        #profile = ps.metrics.representative_elementary_volume(im=im, npoints=5000)
        
        # Compute unique volume values and their corresponding average porosities
        unique_volumes = np.unique(profile.volume)
        avg_porosity = [
            profile.porosity[profile.volume == vol].mean() 
            for vol in unique_volumes
        ]
        # Plot averaged points
        axes[idx].plot(unique_volumes / (resolution ** 2), avg_porosity, 'bo-', label='Averaged Points')
        
        # Original data points for reference (optional)
        axes[idx].plot(profile.volume / (resolution ** 2), profile.porosity, 'r.', alpha=0.3, label='Original Points')
        
        # Labeling
        axes[idx].set_xlabel("Relative Volume")
        axes[idx].set_ylabel("Porosity")
        axes[idx].set_title(f'Plot for i={i}')
        axes[idx].legend()
        
    
    plt.tight_layout()
    plt.show()
'''