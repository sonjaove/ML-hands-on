'''this script is used to interpolate the data from from 0.25 to 0.1 resolution and save the images 
this script will interpolate the values from 0.25 to 0.1 resolution and save the images in the output folder, the methhods here in the class:
1. interpolate_values: this method will interpolate the values from 0.25 to 0.1 using the fed data
2. save_images: this method will save the images in the output folder'''


#importing the required libraries
import xarray as xr
import scipy.interpolate as sci
from tqdm import tqdm
from resize import Resize
import numpy as np
import matplotlib.pyplot as plt
import PIL
import os
import pandas as pd
#from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
import PIL.Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
import torch

process=Resize()
class interpolate():
    
    def make_image(self, data_pth):
        # Load the dataset
        ds = xr.open_dataset(data_pth)
        interpolated_data = np.array(ds['precipitation'].values)
        images = []

        # Loop through each time step and save the image
        for i, data in tqdm(enumerate(interpolated_data), total=len(interpolated_data)):
            fig, ax = plt.subplots()
            
            # Normalize the data for color mapping
            data_normalized = (data - np.min(data)) / (np.ptp(data))
            
            # Plot the data as an image
            cax = ax.imshow(data_normalized, cmap='viridis', interpolation='nearest')
            fig.colorbar(cax)
            
            # Overlay the original data values on the image
            for (j, k), val in np.ndenumerate(data):
                ax.text(k, j, f'{val:.2f}', ha='center', va='center', color='white')
            
            # Remove axes for a cleaner image
            ax.axis('off')
            
            # Save the figure to a PIL image
            fig.canvas.draw()
            image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            images.append(image)
            
            plt.close(fig)  # Close the figure to free up memory

        return images

    
    def interpolate_data(self, nc_file_path, old_resolution, new_resolution):
        # Load the NetCDF dataset
        ds = xr.open_dataset(nc_file_path)
        data_var_name = max(ds.data_vars, key=lambda var: ds[var].size)
        precipitation_data = ds[data_var_name].values

        # Print out the shape of precipitation_data to debug
        print(f"Precipitation data shape: {precipitation_data.shape}")
        
        # Calculate the zoom factor for interpolation (for downscalling, this can also be used for upscalling)
        zoom_factor = old_resolution / new_resolution
        
        # Interpolate the data with cubic spline interpolation
        interpolated_precipitation_data = []
        
        if len(precipitation_data.shape) == 3:
            # Iterate over time steps if there are multiple
            for data in precipitation_data:
                interpolated_data = zoom(data, zoom_factor, order=3)
                # Set negative values to 0
                interpolated_data[interpolated_data < 0] = 0
                interpolated_precipitation_data.append(interpolated_data)
        else:
            # Handle single time step case
            interpolated_precipitation_data = zoom(precipitation_data, zoom_factor, order=3)
            # Set negative values to 0
            interpolated_precipitation_data[interpolated_precipitation_data < 0] = 0
        
        return np.array(interpolated_precipitation_data)


    def save_xarray(self,interpolated_data,data_pth,save_path):	
        # Load the dataset
        ds = xr.open_dataset(data_pth)
        start_date = ds['time'].values[0]
        end_date = ds['time'].values[-1]

        lat = ds['lat'].values
        lon=ds['lon'].values
        inter_lat,inter_lon=np.meshgrid(np.arange(lat.min(),lat.max(),0.1),np.arange(lon.min(),lon.max(),0.1))
        # Create a DataArray from the interpolated data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        interpolated_data_xr = xr.DataArray(
            interpolated_data,
            dims=['time', 'lat', 'lon'],
            coords={'time': date_range, 'lat': inter_lat, 'lon': inter_lon}
            
            )
        
        # Save the interpolated data to a NetCDF file
        interpolated_data_xr.to_netcdf(save_path)

        print(f"Interpolated data saved to {save_path}")
    

    def save_images(self,images, output_folder):
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save each image to the output folder
        for i, image in tqdm(enumerate(images), total=len(images)):
            image.save(os.path.join(output_folder, f'image_{i}.png'))

        print(f"Images saved to {output_folder}")

    #thsi method is not helpign much for the imerge data.
    def resample(self,lr_pth,old_res=0.25,new_res=0.1):
        #we interpolate both the data points to thier respective grids.
        #lr=xr.open_dataset(lr_pth)
        zoom = new_res/old_res
        if zoom<1: #case of upsmapling
            return self.interpolate_data(lr_pth,old_res,new_res)  
        else: #case of downsampling
           return  self.pooling(lr_pth,old_res,new_res)
          
    def pooling(self, data_pth,input_res=0.25,output_res=0.1 ): #will be used for downsampling the images, for now percdr datais used to downsample the data.
        #loding the data
        ds = xr.open_dataset(data_pth)
        data_var_name = max(ds.data_vars, key=lambda var: ds[var].size)
        precipitation_data = ds[data_var_name].values

        #defining the pooling layer
        pooling = torch.nn.AvgPool2d(2,2)
        #convilving the filter on the data
        interpolated_precipitation_data = []
        if len(precipitation_data.shape) == 3:
            # Iterate over time steps if there are multiple
            for data in precipitation_data:
                data = torch.tensor(data).unsqueeze(0).unsqueeze(0)
                interpolated_data = pooling(data).squeeze(0).squeeze(0).numpy()
                interpolated_precipitation_data.append(interpolated_data)
        else:
            # Handle single time step case
            data = torch.tensor(precipitation_data).unsqueeze(0).unsqueeze(0)
            interpolated_precipitation_data = pooling(data).squeeze(0).squeeze(0).numpy()

        return np.array(interpolated_precipitation_data)





    def show_data_with_boundary(self,interpolated_data, num_samples=5, minx=72, miny=29, maxx=80, maxy=37):
        # Extract the precipitation data (assuming it's already interpolated and cropped)
        precipitation = interpolated_data

        print(f"Raw data shape: {precipitation.shape}")
        print(f"Raw data range: [{precipitation.min()}, {precipitation.max()}]")
        print(f"Raw data mean: {precipitation.mean()}")
        print(f"Raw data std: {precipitation.std()}")

        sample_indices = np.linspace(0, len(precipitation) - 1, num_samples, dtype=int)

        fig, axes = plt.subplots(1, num_samples, figsize=(20, 4), subplot_kw={'projection': ccrs.PlateCarree()})
        if num_samples == 1:
            axes = [axes]

        for i, ax in zip(sample_indices, axes):
            # Apply Gaussian blur
            #blurred_data = gaussian_filter(precipitation[i], sigma=1.0)
            
            # Plot the blurred image
            im = ax.imshow(precipitation[i], cmap='viridis', extent=[minx, maxx, miny, maxy], transform=ccrs.PlateCarree())
            ax.set_title(f'Raw - Time step: {i}')
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='red')
            ax.add_feature(cfeature.STATES, linestyle='-', edgecolor='black')  # Add state boundaries if needed
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()