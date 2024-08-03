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
import PIL.Image as Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
import torch

process=Resize()
class interpolate():

    def make_image_from_data(self, data):
        images = []

        for i,j in tqdm(enumerate(data), total=len(data)):
            colormap=plt.get_cmap('viridis')
            rgba_img = colormap(j,bytes=True)[:,:,:3]
            images.append(Image.fromarray(rgba_img, 'RGB'))

        return images

    #this method is used to upsample images.
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
            '''bicubic interpolation -  considers the nearest 4x4 grid of points to estimate new values. 
            This method is smoother and typically produces better results for resizing images or data because it accounts for more neighboring points.'''
            interpolated_precipitation_data = zoom(precipitation_data, zoom_factor, order=3)
            # Set negative values to 0
            interpolated_precipitation_data[interpolated_precipitation_data < 0] = 0
        
        return np.array(interpolated_precipitation_data)
    
    def generate_lat_lon(self, data_pth, new_res=0.1, old_res=0.25):
    # Check if the dataset file exists
        if not os.path.exists(data_pth):
            raise FileNotFoundError(f"The dataset file at {data_pth} does not exist.")
        
        # Load the dataset
        ds = xr.open_dataset(data_pth)
        lat = ds['lat'].values
        lon = ds['lon'].values
        
        # Check the resolution change
        zoom = new_res / old_res
        if zoom <= 1:  # Case of upsampling
            lat_edges = np.arange(lat.min(), lat.max() , new_res)
            lon_edges = np.arange(lon.min(), lon.max(), new_res)
            return lat_edges, lon_edges  # This is case specific as we are generating the hr and lr from the same data
        else:  # Case of downsampling
            # Generate new latitude and longitude edges
            lat_edges = np.arange(lat.min(), lat.max() - old_res, new_res)
            lon_edges = np.arange(lon.min(), lon.max()- old_res, new_res)
            return lat_edges, lon_edges

    def save_xarray(self,interpolated_data,data_pth,save_path,new_res,old_res=0.25):	
        # Load the dataset
        ds = xr.open_dataset(data_pth)
        start_date = ds['time'].values[0]
        end_date = ds['time'].values[-1]

        lat_edges, lon_edges = self.generate_lat_lon(data_pth, new_res,old_res)
        # Create a DataArray from the interpolated data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        expected_shape = (len(date_range), len(lat_edges), len(lon_edges))
        assert interpolated_data.shape == expected_shape, f"Expected shape {expected_shape}, but got {interpolated_data.shape}"

        # Create a DataArray from the interpolated data
        interpolated_data_xr = ds = xr.Dataset(
    {"precipitation": (("time", "lat","lon"), interpolated_data)},
    coords={
        "time": date_range,
        "lat": lat_edges,
        "lon": lon_edges,
    },
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
        else: #case of downsampling, it would change the coords by 0.1 
           return  self.pooling(lr_pth,old_res,new_res)
          
    def pooling(self, data_pth,input_res=0.25,output_res=0.1 ): #will be used for downsampling the images, for now percdr datais used to downsample the data.
        #loding the data
        ds = xr.open_dataset(data_pth)
        data_var_name = max(ds.data_vars, key=lambda var: ds[var].size)
        precipitation_data = ds[data_var_name].values

        #defining the pooling layer
        #this will actually cause the grid to decrease by 0.1 on all sides
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
