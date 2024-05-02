############################################################
# this file has all the necessary functions for question 1.1
############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import lpips
import pickle
from PIL import Image
import os
import shutil
from torch.nn.functional import cosine_similarity
from dataset_utils import *

# load lpips
perceptual_loss = lpips.LPIPS(net='alex')

# load YOLO model
yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("yolo loaded")

# transform for resizing and converting images to tensor
resize_and_norm_transform = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# function for capturing frames and patches
# uses some code from https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/#simple-example
def capture_patches(path, conf_threshold=0.75, reset_cntr=150, mse_threshold=0.15, dark_threshold=25, percep_threshold=0.5, human_patches=[], bb_arr=[], frame_nums=[]):
    
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    transformed = []
    k = 0
    if cap.isOpened():
        print(f"Running YoloV5 on {path}")
        num_added_patches = 0
        cntr = 1
        pbar = tqdm(range(total_frames), desc="Searching for human patches", postfix={"num_patches": len(human_patches), "buffer_size": num_added_patches})
        for k in pbar:
            pbar.set_postfix({"num_patches": len(human_patches), "buffer_size": num_added_patches}, refresh=True)
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects in the frame
            results = yolov5(frame)

            # results contains detected objects and their information (e.g., labels, coordinates)
            # Check if 'person' is among the detected classes
            new_num_added_patches = 0
            if 'person' in list(results.names.values()):
                #print("Human/s detected in frame")
                df = results.pandas().xyxy[0]
                for row in range(len(df)):  # results.xyxy[0] contains bbox coords, confidence, and class
                    confidence, cls, name =  df.loc[row]['confidence'], df.loc[row]['class'], df.loc[row]['name']
                    
                    # Checking if the class is human and that the prediction exceeds confidence threshold
                    if name == 'person' and confidence > conf_threshold:  
                        x1, y1, x2, y2 = map(int, [df.loc[row]['xmin'], df.loc[row]['ymin'], df.loc[row]['xmax'], df.loc[row]['ymax']])
                        width, height = x2 - x1, y2 - y1
                        bb_pixels = width * height
                        human_patch = frame[y1:y2, x1:x2]
                        patch_tf = resize_and_norm_transform(human_patch)
                        
                        # if we haven't added any patches recently then no need to do the MSE check
                        if len(transformed) == 0 or num_added_patches == 0:
                            grey_patch = cv2.cvtColor(human_patch, cv2.COLOR_BGR2GRAY)
                            average_intensity = np.mean(grey_patch)
                            
                            # checks for average intensity and height/width
                            if average_intensity > dark_threshold and (height > 300 and width > 300):
                                human_patches.append(human_patch)
                                transformed.append(patch_tf)
                                frame_nums.append(k) # also append the frame number
                                bb_arr.append((x1, y1, x2, y2))
                                new_num_added_patches += 1
                                cntr = 1
                        
                        # don't keep images that are similar to the previous 5 images in the buffer
                        # I found that combining MSE and perceptual loss was a reliable (but not perfect) way of finding similar images
                        elif return_min_perceptual_loss(transformed[-min(num_added_patches, 5):], patch_tf) > percep_threshold \
                             and return_min_MSE_loss(transformed[-min(num_added_patches, 5):], patch_tf) > mse_threshold:
                            grey_patch = cv2.cvtColor(human_patch, cv2.COLOR_BGR2GRAY)
                            average_intensity = np.mean(grey_patch)
                            
                            # checks for average intensity and height/width
                            if average_intensity > dark_threshold and (height > 300 and width > 300):
                                human_patches.append(human_patch)
                                transformed.append(patch_tf)
                                frame_nums.append(k) # also append the frame number
                                bb_arr.append((x1, y1, x2, y2))
                                new_num_added_patches += 1
                                cntr = 1
                                
                        
            
            # increment number of added patches
            num_added_patches += new_num_added_patches
            
            if cntr % reset_cntr == 0:
                num_added_patches = new_num_added_patches
                cntr = 1
            else:
                cntr += 1
                

            
            del frame, results

    else:
        print("Not running (something broken)")

    # Release the video capture object
    cap.release()
    return human_patches, bb_arr, frame_nums

# save the frames from the video using the frame idxs saved
def save_frames_from_framenums(path_to_video, new_path, frame_idxs):
    cap = cv2.VideoCapture(path_to_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_save = []
    k = 0
    i = 0
    if cap.isOpened():
        print(f"Running YoloV5 on {path_to_video}")
        pbar = tqdm(range(total_frames), desc="Acquiring frames")
        for k in pbar:
            ret, frame = cap.read()
            if k in frame_idxs:
                # resize frame to 256x256
                resized_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                frames_to_save.append(resized_frame)
                cv2.imwrite(f'{new_path}/{i}.jpg', resized_frame)
                i += 1
    cap.release()
    print(f"captured {len(frames_to_save)} frames")
    return frames_to_save


# returns the minimum MSE loss between a single image and an array of images
def return_min_MSE_loss(arr, instance):
    min_loss = 10000
    for p in arr:
        d = F.mse_loss(instance, p)
        if d < min_loss:
            min_loss = d
    return min_loss

# returns the minimum perceptual loss between a single image and an array of images
def return_min_perceptual_loss(arr, instance):
    min_loss = 10000
    for p in arr:
        d = perceptual_loss(instance, p)
        if d < min_loss:
            min_loss = d
    return min_loss


# Copy each file from each source folder to the destination folder
def move_to_folder_and_rename(source_folders, destination_folder, prefixes):
    os.makedirs(destination_folder, exist_ok=True)
    for k in range(len(source_folders)):
        
        omit_files = []
        if "omit.txt" in os.listdir(source_folders[k]): 
            omit_files = add_filenames_to_array(source_folders[k] + "/omit.txt")
        
        for filename in os.listdir(source_folders[k]):
            file_path = os.path.join(source_folders[k], filename)
            
            if os.path.isfile(file_path) and not (file_path in omit_files):
                destination_path = os.path.join(destination_folder, prefixes[k]+filename)
                shutil.copy(file_path, destination_path)
    print("Files have been copied successfully.")