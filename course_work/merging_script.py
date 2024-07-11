'''this file contains the code to merge the data from the downloaded files and crop the data to the required area 
you can change the method for_all_years to suite your needs.
you can also change the lat, lon cordinates to crop the data to your required area.
Set an input folder and get an ouput file. if your folder has subfolders for each year, the code will merge the data for each year and save it in the output file for each year
if not than you'll have to modify the method `for_all_years` to suit your needs, do not mend with the method `preprocess_crop_and_merge`.
(you can also try using the method `merge_netCDF_files`).'''

import xarray as xr
import os
import glob
import numpy as np
import warnings
from xarray.coding.variables import SerializationWarning
import tqdm

def preprocess_crop_and_merge(input_folder, output_file, lat_min=32.324276, lat_max=37.370157, lon_min=72.048340, lon_max=80.727539):
    # PERSIANN-CDR_v01r01_
    file_pattern = os.path.join(input_folder, 'PERSIANN-CDR_v01r01_*.nc')
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        print(f"No files found in {input_folder}")
        return
    
    datasets = []
    for file in files:
        with warnings.catch_warnings():
            # Ignore warnings specific to each dataset load
            warnings.simplefilter("ignore")
            ds = xr.open_dataset(file)
        
        # Clean up fill values and missing values
        if 'precipitation' in ds.data_vars:
            # Ensure that fill values are consistent
            ds['precipitation'].encoding.pop('missing_value', None)
            ds['precipitation'].encoding['_FillValue'] = -9999.0
        
        # Get the indices using np.where and crop the data
        lat_indices = np.where((ds.lat >= lat_min) & (ds.lat <= lat_max))[0]
        lon_indices = np.where((ds.lon >= lon_min) & (ds.lon <= lon_max))[0]
        ds_cropped = ds.isel(lat=lat_indices, lon=lon_indices)
        
        datasets.append(ds_cropped)
    
    # Check if we have datasets to concatenate
    if datasets:
        # Concatenate all datasets along the time dimension
        combined_ds = xr.concat(datasets, dim='time', data_vars='minimal', coords='minimal', compat='override')

        # Handle any potential serialization warnings due to encoding settings when saving
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SerializationWarning)
            combined_ds.to_netcdf(output_file)
        
        # Close datasets to free up resources
        for ds in datasets:
            ds.close()
    else:
        print(f"No datasets to concatenate in {input_folder}")

def for_all_the_years(input_folder='C:\\Users\\Ankit\\Documents\\Vedanshi\\nc_files',start_year=2000,end_year=2024):
    for year in tqdm.tqdm(range(start_year, end_year), desc='Processing files', ascii=True):
        year_folder = os.path.join(input_folder, str(year))
        output_file = rf'C:\Users\Ankit\Documents\Vedanshi\nc_merged\merged_noaa\{year}.nc'
        preprocess_crop_and_merge(year_folder, output_file)

for_all_the_years()

'''this method will merge the data for each year and save it in the output file for the all the years in a single file.
the input folder should contain merged files of a particular year of all the years.


A use case is shown in final_merging.ipynb file.'''

def merge_netCDF_files(input_folder, output_file):
    # Create a list of all NetCDF files sorted by their filename (assuming filenames contain dates)
    file_pattern = os.path.join(input_folder, '*.nc')
    files = sorted(glob.glob(file_pattern))
    
    # Load datasets into a list and concatenate them manually
    datasets = [xr.open_dataset(file) for file in files]
    combined_ds = xr.concat(datasets, dim='time')

    # Save the combined dataset to a new NetCDF file
    combined_ds.to_netcdf(output_file)
    for ds in datasets:  # Close all datasets to free up resources
        ds.close()
    print(f"Saved combined dataset to a single file")

merge_netCDF_files(r'C:\Users\Ankit\Documents\Vedanshi\nc_merged\merged_noaa', r'C:\Users\Ankit\Documents\Vedanshi\nc_merged\CROWN.nc')