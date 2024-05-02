############################################################
# this file has functions shared across all 
############################################################

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.mixture import GaussianMixture
from torch.nn.functional import cosine_similarity
import torch
import shutil

# displays an image
def display_img(img):
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.show()
    

def read_images_into_array(directory_path, omit=None):
    arr = []
    
    if omit != None:
        with open(omit, 'r') as file:
            omitted = [line.strip() for line in file]
        
    for image_file in os.listdir(directory_path):
        if image_file[-1] == 'g' and (directory_path + "/" + image_file) not in omitted:
            arr.append(np.array(Image.open(directory_path + "/" + image_file).convert("RGB")))
    return arr

def read_images_into_array_from_filenames(filenames):
    arr = []
    for image_file in filenames:
        if image_file[-1] == 'g':
            arr.append(np.array(Image.open(image_file).convert("RGB")))
    return arr

def read_images_into_array_from_txt(txt):
    arr = []
    with open(txt, 'r') as file:
        filenames = [line.strip() for line in file]
    
    for image_file in filenames:
        if image_file[-1] == 'g':
            arr.append(np.array(Image.open(image_file).convert("RGB")))
    return arr

# creates a grid of images from an array of numpy images
def create_image_grid(images, grid_size, image_size=(256, 256), bgr2rgb=True):
    # Calculate the grid's width and height 
    grid_width = image_size[0] * grid_size[1]
    grid_height = image_size[1] * grid_size[0]
    
    # Create a new blank image for the grid
    grid_image = Image.new('RGB', (grid_width, grid_height))
    
    # Iterate over the images and paste them into the grid
    for index, img_array in enumerate(images):
        # Convert NumPy array to PIL Image
        if bgr2rgb:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_array)
        img = img.resize(image_size)
        # Calculate the position of this image in the grid
        row = index // grid_size[1]
        col = index % grid_size[1]
        position = (col * image_size[0], row * image_size[1])
        # Paste the image into the grid
        grid_image.paste(img, position)
    
    return grid_image

# takes in a path to a txt file containing a list of files, and returns them as an array
def add_filenames_to_array(file_path):
    filenames = []

    with open(file_path, 'r') as file:
        for line in file:
            filename = line.strip()
            filenames.append(filename)
    
    return filenames

# copies files from array of full file paths into destination
def copy_images_from_array_to_folder(file_paths, destination, prefix=''):
    for k in range(len(file_paths)):
        src_file = file_paths[k]
        dst_file = f'{destination}/{prefix}{k}.jpg'
        shutil.copy(src_file, dst_file)



