'''this file contains the code to merge the data from the downloaded files and crop the data to the required area 
you can change the method for_all_years to suite your needs.
you can also change the lat, lon cordinates to crop the data to your required area.
Set an input folder and get an ouput file. if your folder has subfolders for each year, the code will merge the data for each year and save it in the output file for each year
if not than you'll have to modify the method `for_all_years` to suit your needs, do not mend with the method `preprocess_crop_and_merge`.
(you can also try using the method `merge_netCDF_files`).'''


#3B-DAY.MS.MRG.3IMERG.20000606-S000000-E235959.V07B.nc4.nc4
#3B-DAY.MS.MRG.3IMERG.20000609-S000000-E235959.V07B.nc4.nc4

import xarray as xr
import os
import glob
import numpy as np
import warnings
from xarray.coding.variables import SerializationWarning
import tqdm

def preprocess_crop_and_merge(input_folder, output_file, lat_min=32, lat_max=24, lon_min=72, lon_max=80):
    # PERSIANN-CDR_v01r01_
    file_pattern = os.path.join(input_folder, 'IMERG******precip.nc4')
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


def for_all_the_years(input_folder='C:\\Users\\Ankit\\Documents\\Vedanshi\\TRMM_Kumar',start_year=2013,end_year=2024):
    for year in tqdm.tqdm(range(start_year, end_year), desc='Processing files', ascii=True):
        year_folder = os.path.join(input_folder, str(year))
        output_file = rf'C:\Users\Ankit\Documents\Vedanshi\nc_merged\{year}.nc4'
        preprocess_crop_and_merge(year_folder, output_file)

#for_all_the_years()

'''this method will merge the data for each year and save it in the output file for the all the years in a single file.
the input folder should contain merged files of a particular year of all the years.


A use case is shown in final_merging.ipynb file.'''

def merge_netCDF_files(input_folder, output_file,lat_min=32, lat_max=24, lon_min=72, lon_max=80):
    # Create a list of all NetCDF files sorted by their filename (assuming filenames contain dates)
    file_pattern = os.path.join(input_folder, 'IMERG******precip.nc4')
    files = sorted(glob.glob(file_pattern))

    datasets = []
    for file in tqdm.tqdm(files, desc='Processing files', ascii=True):
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


merge_netCDF_files(r'F:\TANISHQ\IMERG_DATA', r'F:\TANISHQ\IMERG_DATA\CROWN_GT.nc4')