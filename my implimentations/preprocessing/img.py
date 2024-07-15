import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
from torch.utils.data import Dataset
#import torch
import torchvision.transforms as transforms
import os

#this class is used to preprocess the images
class image():
    def __init__(self):
        pass
    def add_images_from_directory(self,directory):
        images = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.jpg', '.png')):  # Add more extensions if needed
                image_path = os.path.join(directory, filename)
                try:
                    with Image.open(image_path) as img:
                        images.append(img.copy())  # We use copy() to keep the image in memory after closing the file
                except IOError:
                    print(f"Error loading image: {image_path}")
        return images
    def blur_image(self,inp_dir,out_dir,rad=1):
        for filename in os.listdir(inp_dir):
            if filename.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(inp_dir, filename)
                try:
                    with Image.open(image_path) as img:
                        img=img.filter(ImageFilter.GaussianBlur(rad))
                        img.save(os.path.join(out_dir,filename))
                except IOError:
                    print(f"Error loading image: {image_path}")
        print("Done")
    def unblur_image(self,inp_dir,out_dir,rad=1):
        for filename in os.listdir(inp_dir):
            if filename.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(inp_dir, filename)
                try:
                    with Image.open(image_path) as img:
                        img=img.filter(ImageFilter.UnsharpMask(rad))
                        img.save(os.path.join(out_dir,filename))
                except IOError:
                    print(f"Error loading image: {image_path}")
        print("Done")
    #we would also have to add a method that gives the corresponding hr images from the lr images using gaussian blur anf stuff.
    def lr_hr(self,lr_directory,hr_directory,ycbcr=False):
        self.lr=[]
        self.lr=self.add_images_from_directory(lr_directory)
        self.hr=self.add_images_from_directory(hr_directory)
        if ycbcr:
            lr_ycbcr=[image.convert('YCbCr') for image in self.lr]
            hr_ycbcr=[image.convert('YCbCr') for image in self.hr]
            lr_y=[np.array(image)[:,:,0] for image in lr_ycbcr]
            hr_y=[np.array(image)[:,:,0] for image in hr_ycbcr]
            self.lr=lr_y
            self.hr=hr_y

        return self.lr,self.hr
    
    
#this class is used to create the dataset for the model
class SRDataset(Dataset):
    def __init__(self, lr_images, hr_images, transform=None):
        self.lr_images = lr_images
        self.hr_images = hr_images
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image = self.lr_images[idx]
        hr_image = self.hr_images[idx]
        
        # Convert PIL Images to tensors
        to_tensor = transforms.ToTensor()
        lr_tensor = to_tensor(lr_image)
        hr_tensor = to_tensor(hr_image)
        
        if self.transform:
            lr_tensor = self.transform(lr_tensor)
            hr_tensor = self.transform(hr_tensor)
        
        return lr_tensor, hr_tensor
#img=image()
#img.unblur_image(r"C:\Users\Ankit\Documents\Vedanshi\T_91\T91_LR",r"C:\Users\Ankit\Documents\Vedanshi\T_91\T91_LR_unblur",rad=10)
#img2=image()
#img2.unblur_image(r"C:\Users\Ankit\Documents\Vedanshi\T_91\T91_HR",r"C:\Users\Ankit\Documents\Vedanshi\T_91\T91_HR_unblur",rad=3)