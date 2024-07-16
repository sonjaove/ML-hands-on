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
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
import PIL.Image

process=Resize()
class Interpolate:
    def getting_points(self, data_pth):

        # input_res points(the space where data exists).

        ds = xr.open_dataset(data_pth)

        df=ds.to_dataframe().reset_index()

        #if df.columns in ['nv','lon_bnds','lat_bnds']:
        df.drop(columns=['nv','lon_bnds','lat_bnds'],inplace=True)

        if df.isnull().sum().sum()>0:
            df.fillna(0,inplace=True)
        
        df.drop_duplicates(inplace=True,ignore_index=True)

        days=len(df['time'].unique())
        step=len(df['lat'].unique())*len(df['lon'].unique())
        first_day=df.iloc[0:step]
        lat=np.array(first_day['lat'])
        lon=np.array(first_day['lon'])
        prcp_points = np.column_stack((lon.ravel(), lat.ravel()))
        prcp=np.array(first_day['precipitation'])
        print(len(prcp_points)==len(prcp))
        #output_res points
        #these are the centroids of the grid cells for the output resolution grid.
        inter_lon,inter_lat=process.res_change_general(min(lon), min(lat), max(lon), max(lat), input_res=0.25, output_res=0.1)
        lon_rav=np.array(inter_lon).ravel()
        lat_rav=np.array(inter_lat).ravel()
        inter_points=np.column_stack((lon_rav,lat_rav))
        inter_points=pd.DataFrame(inter_points,columns=['lon','lat'])
        inter_points.drop_duplicates(inplace=True)	
        inter_points
        
        unique_lats = np.sort(inter_points['lat'].unique())
        unique_lons = np.sort(inter_points['lon'].unique())

        return inter_lon,inter_lat
    
    def visualize_res_change(self, data_pth):
        ds=xr.open_dataset(data_pth)
        process.res_change(ds['lon'].min(), ds['lat'].min(), ds['lon'].max(), ds['lat'].max(), input_res=0.25, output_res=0.1)
    
    def show_raw_data( self,data_pth, num_samples=5):
        ds = xr.open_dataset(data_pth)

        precipitation = ds['data'].values

        print(f"data shape: {precipitation.shape}")
        print(f"data range: [{precipitation.min()}, {precipitation.max()}]")
        print(f"data mean: {precipitation.mean()}")
        print(f"data std: {precipitation.std()}")

        sample_indices = np.linspace(0, precipitation.shape[0] - 1, num_samples, dtype=int)

        fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
        if num_samples == 1:
            axes = [axes]

        for i, ax in zip(sample_indices, axes):
            im = ax.imshow(precipitation[i], cmap='viridis')
            ax.set_title(f'Raw - Time step: {i}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()    

obj=Interpolate()

class image_interpolation():
    
    
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
    
    
    def interpolate_precipitation_data(self, nc_file_path, old_resolution, new_resolution):
        # Load the NetCDF dataset
        ds = xr.open_dataset(nc_file_path)
        
        # Extract precipitation data
        precipitation_data = ds['precipitation'].values
        
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
    

    def show_images(self, interpolated_data, num_samples=5):

        # Get the precipitation values from the dataset
        #precipitation_values = dataset['data'].values
        sample_indices = np.linspace(0, len(interpolated_data) -1, num_samples, dtype=int)
        
        # Display the images
        fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
        if num_samples == 1:
            axes = [axes]

        for i, idx in enumerate(sample_indices):
            ax = axes[i]
            im = ax.imshow(interpolated_data[idx], cmap='viridis')
            ax.set_title(f'Image {idx}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04 )

        plt.tight_layout()
        plt.show()


    def save_images(self,images, output_folder):
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save each image to the output folder
        for i, image in tqdm(enumerate(images), total=len(images)):
            image.save(os.path.join(output_folder, f'image_{i}.png'))

        print(f"Images saved to {output_folder}")
