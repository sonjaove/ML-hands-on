''' these is the dump from pasthistory.ipynb file.'''

# using minx and maxx, i.e the boundary poits for the grid.
# initialize an empty list to store x-values for the custom grid
x_values = []
# generate x-values
current_x = minx
increment = 0.25 #this is an inceremnet of 0.25
while current_x <= maxx:
    x_values.append(current_x)  
    current_x += increment
len(x_values)

# using miny and maxy, i.e the boundary poits for the grid.
# initialize an empty list to store y-values for the custom grid
y_values = []
# generate y-values
current_y = miny
increment = 0.25 #this is an inceremnt of 0.25
while current_y <= maxy:
    y_values.append(current_y)  
    current_y += increment
len(y_values)

x_values = np.array(x_values, dtype=float) # x_values are the longitudes and y_values are the latitudes.
y_values = np.array(y_values, dtype=float)

if __name__ == "__main__":
    grid_cell = initialize_and_populate_grid(x_values, y_values)

grid_cell

xc=[i['Xc']for i in grid_cell]
yc=[i['Yc']for i in grid_cell]
x,y=np.meshgrid(xc,yc)
rectangle = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs='epsg:4326')
fig, ax = plt.subplots()
ax.scatter(x, y, s=1, color='blue')
rectangle.plot(ax=ax, color='none', edgecolor='red')
plt.show()

def res_change(minx,miny,maxx,maxy,input_res,output_res):  
    
        # using minx and maxx, i.e the boundary poits for the grid.
    # initialize an empty list to store x-values for the custom grid
    x_values = []
    # generate x-values
    current_x = minx
    increment = input_res #this is an inceremnet of 0.25
    while current_x <= maxx:
        x_values.append(current_x)  
        current_x += increment
    len(x_values)
    # using miny and maxy, i.e the boundary poits for the grid.
    # initialize an empty list to store y-values for the custom grid
    y_values = []
    # generate y-values
    current_y = miny
    increment = input_res #this is an inceremnt of 0.25
    while current_y <= maxy:
        y_values.append(current_y)  
        current_y += increment
    len(y_values)
    x_values = np.array(x_values, dtype=float) # x_values are the longitudes and y_values are the latitudes.
    y_values = np.array(y_values, dtype=float)
    if __name__ == "__main__":
     grid_cell = initialize_and_populate_grid(x_values, y_values)
    xc=[i['Xc']for i in grid_cell]
    yc=[i['Yc']for i in grid_cell]
    x,y=np.meshgrid(xc,yc)
    rectangle = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs='epsg:4326')
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=1, color='blue')
    rectangle.plot(ax=ax, color='none', edgecolor='red')
    #ax.title('image for input resolution')
    plt.show()


    x_values_output = []
    # generate x-values
    current_x_output = minx
    increment = output_res #this is an inceremnet of 0.25
    while current_x_output <= maxx:
        x_values_output.append(current_x_output)  
        current_x += increment
    len(x_values_output)
    # using miny and maxy, i.e the boundary poits for the grid.
    # initialize an empty list to store y-values for the custom grid
    y_values_output = []
    # generate y-values
    current_y_output = miny
    increment = output_res #this is an inceremnt of 0.25
    while current_y_output <= maxy:
        y_values_output.append(current_y_output)  
        current_y_output += increment
    len(y_values_output)
    x_values_output = np.array(x_values_output, dtype=float) # x_values are the longitudes and y_values are the latitudes.
    y_values_output = np.array(y_values_output, dtype=float)
    if __name__ == "__main__":
     grid_cell_output = initialize_and_populate_grid(x_values_output, y_values_output)
    xc_output=[i['Xc']for i in grid_cell_output]
    yc_output=[i['Yc']for i in grid_cell_output]
    x_out,y_out=np.meshgrid(xc_output,yc_output)
    rectangle_out = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs='epsg:4326')
    fig_out, ax_out = plt.subplots()
    ax_out.scatter(x_out, y_out, s=1, color='black')
    rectangle_out.plot(ax=ax, color='none', edgecolor='red')
    #ax_out.title('image for output resolution')
    plt.show()

if __name__ == "__main__":
    res_change(minx, miny, maxx, maxy, 0.25, 0.1)


# a small trick i wa trying to do, to get all the data for all the years together.

#repeating the lat, lon for 40 years.
start_date = '1984-01-01'
end_date = '2023-12-31'
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
dates_df = pd.DataFrame({'date': date_range})
n_dates = len(date_range)
df_rep = pd.concat([inter_points]*n_dates, ignore_index=True)
df_rep['date']=np.tile(date_range, len(inter_points))
df_rep = df_rep[['date', 'lon', 'lat']]
df_rep


# interpolation using a diffrent class from scipy

def interpolate(initial_res_points, data, coords, method_used):
    # Ensure the entered coordinates have accuracy of only 1 digit after the decimal.
    if method_used in ['linear', 'nearest', 'cubic']:
        interpolator = sci.RegularGridInterpolator(initial_res_points, data, method=method_used)
        return interpolator(coords)
    else:
        raise ValueError("Invalid method. Supported methods are 'linear', 'nearest', and 'cubic'.")
    
# from interpolating.py
def save_images(self, interpolated_data, output_folder):
    '''this method will save the images in the output folder'''
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through each time step and save the image
    for i, data in tqdm(enumerate(interpolated_data), total=len(interpolated_data)):
        # Normalize the data to be in the range 0-255
        data_normalized = (255 * (data - np.min(data)) / (np.ptp(data))).astype(np.uint8)
        
        # Create an Image object from the data
        image = PIL.Image.fromarray(data_normalized)
        #showing the image
        image
        # Save the image
        image.save(os.path.join(output_folder, f'image_{i}.png'))

#fitting the basin/map in the grid, taken from resize.py
    def fitting_grid(self,basin_file, bound_box:tuple, input_res=0.25, output_res=0.1):
        basin=gpd.read_file(basin_file)
        box=basin.total_bounds
        bound_box+=input_res
        dif=box-bound_box
        if dif!=0:
            for idx, difs in enumerate(dif):
                if dif > 0:
                    bound_box[idx] -= difs
                else:
                    bound_box[idx] += difs
        if __name__ == "__main__":
            xc_inp,yc_inp,xc_out,yc_out,grids=self.res_change(bound_box[0], bound_box[1], bound_box[2], bound_box[3], input_res, output_res,crs=basin.crs.to_epsg())
        grid=[]
        for cell in grids:
            polygon = shapely.geometry.Polygon(list(zip(cell['X'], cell['Y'])))
            grid.append(polygon)

        grid_gdf = gpd.GeoDataFrame({'geometry': grid}, crs=basin.crs)
        fig, ax = plt.subplots()
        ax.scatter(xc_out, yc_out, s=1, color='black')
        grid_gdf.boundary.plot(ax=ax, color='blue', label='grid cells')
        basin.boundary.plot(ax=ax, color='red',label='basin')
        plt.legend()
        plt.show()



import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Step 1: Create viridis plot from xarray using PIL instead of matplotlib
def plot_xarray_with_pil(xr_data, save_path):
    data_array = xr_data.values
    # Normalize data for image representation
    normalized_data = (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))
    img = Image.fromarray(np.uint8(plt.cm.viridis(normalized_data)*255))
    img.save(save_path)



# Step 2 & 3: Convert xarray to PyTorch tensor and create a custom Dataset using PIL 
class XarrayDataset(Dataset):
    def __init__(self, xr_data):
        self.data = torch.tensor(xr_data.values, dtype=torch.float32)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]

    # Step 4: Create a DataLoader
    def create_dataloader(xr_data, batch_size=32):
        dataset = XarrayDataset(xr_data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Step 5: Save the DataLoader
    def save_dataloader(dataloader, filepath):
        torch.save(dataloader.dataset, filepath)

    # Function to load the saved DataLoader
    def load_dataloader(filepath, batch_size=32):
        dataset = torch.load(filepath)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Main process
    def process_xarray_with_pil(xr_data, save_path, batch_size=32, image_save_path='viridis_plot.png'):
        # Plot the xarray data using PIL and save the image
        plot_xarray_with_pil(xr_data, image_save_path)
        
        # Create and save DataLoader
        dataloader = create_dataloader(xr_data, batch_size)
        save_dataloader(dataloader, save_path)
        
        print(f"DataLoader saved to {save_path}")
        print(f"Viridis plot saved to {image_save_path}")
        
        # Demonstrate loading
        loaded_dataloader = load_dataloader(save_path, batch_size)
        print("DataLoader successfully loaded")
        
        return loaded_dataloader

    # Usage
    if __name__ == "__main__":
        # Assuming you have an xarray DataArray named 'xr_data'
        # xr_data = xr.DataArray(...)
        
        save_path = 'path/to/save/dataloader.pth'
        image_save_path = 'path/to/save/viridis_plot.png'
        batch_size = 32
        
        loaded_dataloader = process_xarray_with_pil(xr_data, save_path, batch_size, image_save_path)
        
        # You can now use the loaded_dataloader in your PyTorch training loop



#from interpoalting.py

def interpolate_and_create_image(self,data, old_resolution, new_resolution):
        # Calculate the zoom factor for interpolation
        zoom_factor = old_resolution / new_resolution

        # Interpolate the data
        interpolated_data = zoom(data, zoom_factor, order=3)  # order=3 for cubic interpolation

        # Normalize the interpolated data to be in the range 0-255
        data_normalized = (255 * (interpolated_data - np.min(interpolated_data)) / np.ptp(interpolated_data)).astype(np.uint8)

        # Create an Image object from the normalized data
        image = PIL.Image.fromarray(data_normalized)

        return image
    
#from interpolating.py

def downscale_images(self,images, old_resolution, new_resolution):
        downscaled_images = []
        
        for image in images:
            # Convert the image to a numpy array
            data = np.array(image)
            
            # Calculate the zoom factor for downsampling
            zoom_factor = old_resolution / new_resolution

            # Downscale the data using cubic spline interpolation
            if len(data.shape) == 3:  # For RGB images
                downscaled_data = np.stack([zoom(data[:, :, i], zoom_factor, order=3) for i in range(data.shape[2])], axis=2)
            else:  # For grayscale images
                downscaled_data = zoom(data, zoom_factor, order=3)

            # Convert downscaled data back to uint8 (optional step)
            downscaled_data = downscaled_data.astype(np.uint8)

            # Create an Image object from the downscaled data
            downscaled_image = PIL.Image.fromarray(downscaled_data)
            
            # Append the downscaled image to the list
            downscaled_images.append(downscaled_image)
        
        return downscaled_images




## from checkpoints.ipynb
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def visualize_lr_hr_pair(lr_image, hr_image, num_pairs=3):
    # Convert tensors to PIL images
    to_pil = transforms.ToPILImage()
    
    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 5*num_pairs))
    
    for i in range(num_pairs):
        if i < len(lr_image):
            # Display LR image
            axes[i, 0].imshow(to_pil(lr_image[i]))
            axes[i, 0].set_title(f'LR Image {i+1}')
            axes[i, 0].axis('off')
            
            # Display HR image
            axes[i, 1].imshow(to_pil(hr_image[i]))
            axes[i, 1].set_title(f'HR Image {i+1}')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


# Example usage
from dataloader_1 import process_images
if __name__ == "__main__":
    # Example with directories
    lr_input = images
    hr_input = interpolated_images
    save_path = r"C:\Users\Ankit\Documents\Vedanshi\ML-hands-on\course_work\crown\data"
    batch_size = 64
    tts_ratio = 0.9

    train_loader, test_loader = process_images(lr_input, hr_input, save_path, batch_size=batch_size, tts_ratio=tts_ratio)

