from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from PIL import Image, ImageFilter
import os
import torch
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, lr_input, hr_input, transform=None):
        if isinstance(lr_input, str) and isinstance(hr_input, str):
            # Input is directories
            self.lr_files = [os.path.join(lr_input, f) for f in os.listdir(lr_input)]
            self.hr_files = [os.path.join(hr_input, f) for f in os.listdir(hr_input)]
        elif isinstance(lr_input, list) and isinstance(hr_input, list):
            # Input is lists of images
            self.lr_files = lr_input
            self.hr_files = hr_input
        else:
            raise ValueError("Inputs should either be directories or lists of images.")

        self.transform = transform

    def __len__(self):
        return min(len(self.lr_files), len(self.hr_files))

    def __getitem__(self, idx):
        if isinstance(self.lr_files[idx], str) and isinstance(self.hr_files[idx], str):
            lr_image = Image.open(self.lr_files[idx]).convert('RGB')
            hr_image = Image.open(self.hr_files[idx]).convert('RGB')
        else:
            lr_image = self.lr_files[idx].convert('RGB')
            hr_image = self.hr_files[idx].convert('RGB')

        #lr_image = lr_image.filter(ImageFilter.GaussianBlur(radius=1))

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        else:
            # Convert PIL images to tensors
            lr_image = torch.tensor(np.array(lr_image), dtype=torch.float32).permute(2, 0, 1)  # CxHxW
            hr_image = torch.tensor(np.array(hr_image), dtype=torch.float32).permute(2, 0, 1)  # CxHxW
        
        return lr_image, hr_image

    def split_train_test_indices(self, split_ratio=0.7, shuffle=True, random_seed=42):
        dataset_size = len(self)
        indices = list(range(dataset_size))
        split = int(np.floor(split_ratio * dataset_size))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_indices, test_indices = indices[:split], indices[split:]

        return train_indices, test_indices

    def get_train_loader(self, batch_size=64, num_workers=4, split_ratio=0.7, shuffle=True, random_seed=42):
        train_indices, _ = self.split_train_test_indices(split_ratio, shuffle, random_seed)
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        return train_loader

    def get_test_loader(self, batch_size=64, num_workers=4, split_ratio=0.7, shuffle=True, random_seed=42):
        _, test_indices = self.split_train_test_indices(split_ratio, shuffle, random_seed)
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = DataLoader(self, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
        return test_loader

    def save_dataloader(self, dataloader, filepath):
        torch.save(dataloader.dataset, filepath)

    def load_dataloader(filepath, batch_size=32, shuffle=True):
        dataset = torch.load(filepath)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image_np = np.array(image)
        
        if image_np.ndim == 2:  # Grayscale image
            image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        else:  # RGB image
            image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)
        
        return image_tensor

    def seek(self, idx):
        if idx < 0 or idx >= len(self.images):
            raise IndexError("Index out of range")

    def get_current_index(self):
        return self.current_idx

    
def create_dataloader(images, batch_size=32, shuffle=True):
    dataset = ImageDataset(images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def save_dataloader(dataloader, filepath):
    torch.save(dataloader.dataset, filepath)

def load_dataloader(filepath, batch_size=32, shuffle=True):
    dataset = torch.load(filepath)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def split_train_test(images, train_split=0.8):
    num_images = len(images)
    indices = list(range(num_images))
    split = int(np.floor(train_split * num_images))
    np.random.shuffle(indices)
    
    train_indices, test_indices = indices[:split], indices[split:]
    train_images = [images[i] for i in train_indices]
    test_images = [images[i] for i in test_indices]
    
    return train_images, test_images

def flatten_nested_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def kfold_validation(lr_images, hr_images,k=5, batch_size=32):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    folds = list(kf.split(lr_images, hr_images))
    
    for fold, (train_indices, test_indices) in enumerate(tqdm(folds, desc='K-Fold Validation')):
        print(f"Processing Fold {fold + 1}/{k}")
        
        # Split dataset into train and test images for this fold
        train_images = [(lr_images,hr_images)[i] for i in train_indices]
        test_images = [(lr_images,hr_images)[i] for i in test_indices]

        # Create Datasets and DataLoaders
        train_dataset = ImageDataset(train_images)
        test_dataset = ImageDataset(test_images)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Save both train and test datasets into a single file
        fold_data = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset
        }
        torch.save(fold_data, f"fold_{fold + 1}.pth")

        print(f"Fold {fold + 1} complete.\n")

    print("K-Fold data prep complete.")

def load_fold_data(filepath):
    fold_data = torch.load(filepath)
    train_dataset = fold_data['train_dataset']
    test_dataset = fold_data['test_dataset']
    return train_dataset, test_dataset

def process_images(lr_input, hr_input, save_path, batch_size=32, k_fold=False, k=5,tts_ratio=0.8):
    # Create CustomDataset
    dataset = CustomDataset(lr_input, hr_input)

    if k_fold:
        # Perform k-fold validation
        kfold_validation(dataset.lr_files, dataset.hr_files, k=k, batch_size=batch_size)
    else:
        # Split dataset into train and test sets
        
        train_indices, test_indices = dataset.split_train_test_indices(split_ratio=tts_ratio)
        
        # Create train DataLoader
        train_sampler = SubsetRandomSampler(train_indices)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        
        # Create test DataLoader
        test_sampler = SubsetRandomSampler(test_indices)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        
        # Save train DataLoader
        train_save_path = save_path + '_train.pth'
        torch.save(train_dataloader.dataset, train_save_path)
        print(f"Train DataLoader saved to {train_save_path}")
        
        # Save test DataLoader
        test_save_path = save_path + '_test.pth'
        torch.save(test_dataloader.dataset, test_save_path)
        print(f"Test DataLoader saved to {test_save_path}")