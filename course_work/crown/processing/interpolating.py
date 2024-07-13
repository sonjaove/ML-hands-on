'''this script is used to interpolate the data from from 0.25 to 0.1 resolution and save the images 
this script will interpolate the values from 0.25 to 0.1 resolution and save the images in the output folder, the methhods here in the class:
1. interpolate_values: this method will interpolate the values from 0.25 to 0.1 using the fed data
2. save_images: this method will save the images in the output folder'''
#importing the required libraries
import xarray as xr
import scipy.interpolate as sci
import tqdm
from resize import Resize
import numpy as np
import matplotlib.pyplot as plt
import PIL
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

process=Resize()
class Interpolate:
    def interpolate_values(self, data_pth):

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
        xc_inp,yc_inp,xc_out,yc_out,grids=process.res_change(min(lat), min(lon), max(lat), max(lon), input_res=0.25, output_res=0.1)
        lon_out=list(np.unique(xc_out[0]))
        lat_out=[]
        for i in yc_out:
            lat_out.append(i[0])
        lat_out=list(np.unique(lat_out))
        #these are the centroids of the grid cells for the output resolution grid.
        inter_lat,inter_lon=np.meshgrid(lat_out,lon_out)
        inter_points=np.column_stack((inter_lon.ravel(),inter_lat.ravel()))
        inter_points=pd.DataFrame(inter_points,columns=['lon','lat'])
        inter_points.drop_duplicates(inplace=True)	
        inter_points
        
        def interpolate(initial_res_points,data,coords,method_used):
            coords = np.array(coords)
            if coords.ndim == 1:
                coords = coords[np.newaxis, :]
            if method_used=='linear' or method_used=='nearest':
                if method_used=='nearest':
                    return sci.griddata(initial_res_points,data,coords,method='nearest') #nearest neighbour
                else:
                    return sci.griddata(initial_res_points,data,coords,method='linear') #linear.
            else :
                    return sci.griddata(initial_res_points,data,coords,method='cubic') #splines.
            
        start_date = '2000-01-01'
        end_date = '2023-12-31'
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        data = xr.DataArray(
            np.zeros((len(date_range), len(lon_out), len(lat_out))), 
            coords={'time': date_range,  'lon': lon_out,'lat': lat_out},
            dims=['time', 'lon', 'lat']
        )
        #CONVERTING THE DATA TO DATAFRAME FOR EASY MANIPULATION.
        df_interpolated=data.to_dataframe(name='data').reset_index()
        step_out=len(df_interpolated['lat'].unique())*len(df_interpolated['lon'].unique())
        #interpolating the values
        for i in tqdm.tqdm(range(days), desc=f'running the loop for {days//365} years', ascii=True):
            prcp=np.array(df.iloc[i*step:(i+1)*step]['precipitation'])
            interpolated_data = interpolate(prcp_points, prcp, inter_points, method_used='nearest')
            df_interpolated['data'][i*step_out:(i+1)*step_out] = interpolated_data
        df_interpolated = df_interpolated.set_index(['time', 'lat', 'lon']).to_xarray()
        nc_file_path = r'C:\Users\Ankit\Documents\Vedanshi\nc_merged\CROWN_Interpolated.nc'
        df_interpolated.to_netcdf(nc_file_path)

        print(f"Dataset saved to {nc_file_path}")
    
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