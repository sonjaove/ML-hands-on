import xarray as xr
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Step 1: Create viridis plot from xarray
def plot_xarray(xr_data):
    plt.figure(figsize=(10, 8))
    xr_data.plot.imshow(cmap='viridis')
    plt.title('Viridis Plot of Xarray Data')
    plt.show()

# Step 2 & 3: Convert xarray to PyTorch tensor and create a custom Dataset
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
def process_xarray(xr_data, save_path, batch_size=32):
    # Plot the xarray data
    plot_xarray(xr_data)
    
    # Create and save DataLoader
    dataloader = create_dataloader(xr_data, batch_size)
    save_dataloader(dataloader, save_path)
    
    print(f"DataLoader saved to {save_path}")
    
    # Demonstrate loading
    loaded_dataloader = load_dataloader(save_path, batch_size)
    print("DataLoader successfully loaded")
    
    return loaded_dataloader

# Usage
if __name__ == "__main__":
    # Assuming you have an xarray DataArray named 'xr_data'
    # xr_data = xr.DataArray(...)
    
    save_path = 'path/to/save/dataloader.pth'
    batch_size = 32
    
    loaded_dataloader = process_xarray(xr_data, save_path, batch_size)
    
    # You can now use the loaded_dataloader in your PyTorch training loop