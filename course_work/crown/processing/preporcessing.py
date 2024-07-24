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

process=Resize()
class interpolate():
    
    def make_image(self,data_pth):
        # Load the dataset
        ds = xr.open_dataset(data_pth)
        interpolated_data = np.array(ds['precipitation'].values)
        images = []

        # Loop through each time step and save the image
        for i, data in tqdm(enumerate(interpolated_data), total=len(interpolated_data)):
            # Normalize the data to be in the range 0-255
            data_normalized = (255 * (data - np.min(data)) / (np.ptp(data))).astype(np.uint8)

            # Create an Image object from the data
            image = PIL.Image.fromarray(data_normalized)
            
            # Add the image to the list
            images.append(image)

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

    def resample(self,lr_pth,hr_pth,minx=72, miny=29, maxx=80, maxy=37,input_res=0.25,output_res=0.1):
        #we interpolate both the data points to thier respective grids.
        lr=xr.open_dataset(lr_pth)
        hr=xr.open_dataset(hr_pth)

        #getting the data.
        data_hr_name = max(hr.data_vars, key=lambda var: hr[var].size)
        data_lr_name = max(lr.data_vars, key=lambda var: lr[var].size)
        x_lr_og=lr['lon'].values
        y_lr_og=lr['lat'].values
        x_hr_og=hr['lon'].values
        y_hr_og=hr['lat'].values
        x_og,y_og=np.meshgrid(x_lr_og,y_lr_og)
        points_25=np.column_stack((x_og.ravel(),y_og.ravel()))
        x_new,y_new=np.meshgrid(x_hr_og,y_hr_og)
        points_10=np.column_stack((x_new.ravel(),y_new.ravel()))

        precipitation_hr = hr[data_hr_name].values
        precipitation_lr = lr[data_lr_name].values    

        #making the grid points.(directly meshgrided points form the method)
        (x_lr,y_lr),(x_hr,y_hr)=process.res_change_general(minx,miny,maxx,maxy,input_res,output_res)
        xi_25=np.column_stack((x_lr.ravel(),y_lr.ravel()))
        xi_10=np.column_stack((x_hr.ravel(),y_hr.ravel()))

        print("Shapes:")
        print("points_25:", points_25.shape)
        print("points_10:", points_10.shape)
        print("precipitation_lr:", precipitation_lr.shape)
        print("precipitation_hr:", precipitation_hr.shape)
        print("xi_25:", xi_25.shape)
        print("xi_10:", xi_10.shape)

        #performing the interpolation
        lr_data=[]
        hr_data=[]
        for i in range(len(precipitation_lr)):
            lr_data.append(sci.griddata(points_25,precipitation_lr[i].ravel(),xi_25,'nearest'))
        for i in range(len(precipitation_hr)):
            hr_data.append(sci.griddata(points_10,precipitation_hr[i].ravel(),xi_10,'nearest'))
        
        lr_data=np.array(lr_data)
        hr_data=np.array(hr_data)

        return lr_data,hr_data

    def show_data_with_boundary(self,interpolated_data, num_samples=5, minx=72, miny=29, maxx=80, maxy=37):
        # Extract the precipitation data (assuming it's already interpolated and cropped)
        precipitation = np.array(interpolated_data)

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
            blurred_data = gaussian_filter(precipitation[i], sigma=1.0)
            
            # Plot the blurred image
            im = ax.imshow(blurred_data, cmap='viridis', extent=[minx, maxx, miny, maxy], transform=ccrs.PlateCarree())
            ax.set_title(f'Raw - Time step: {i} (Blurred)')
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='red')
            ax.add_feature(cfeature.STATES, linestyle='-', edgecolor='black')  # Add state boundaries if needed
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
