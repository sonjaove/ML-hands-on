import xarray as xr
import os
import glob
import numpy as np
import warnings
from xarray.coding.variables import SerializationWarning
import tqdm

def preprocess_crop_and_merge(input_folder, output_file):
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
        lat_indices = np.where((ds.lat >= 25) & (ds.lat <= 30))[0]
        lon_indices = np.where((ds.lon >= 83) & (ds.lon <= 90))[0]
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

def for_all_the_years(input_folder='C:\\Users\\Ankit\\Documents\\Vedanshi\\nc_files'):
    for year in tqdm.tqdm(range(1983, 2024), desc='Processing files', ascii=True):
        year_folder = os.path.join(input_folder, str(year))
        output_file = rf'C:\Users\Ankit\Documents\Vedanshi\nc_merged\merged_noaa\{year}.nc'
        preprocess_crop_and_merge(year_folder, output_file)

for_all_the_years()
