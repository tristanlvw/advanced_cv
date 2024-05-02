############################################################
# this file has all the necessary functions for question 1.2
############################################################

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.functional import cosine_similarity
import torch
import pickle
import json
from dataset_utils import *

# gets the nth frame in an array
def get_nth_frame(path, n):
    cap = cv2.VideoCapture(path)
    k = 0
    if cap.isOpened():
        for k in tqdm(range(n+1)):
            ret, frame = cap.read()
    return frame

# from https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/2109
JOINT_PAIRS_MAP_ALL = {(0, 15): {'joint_names': ('Nose', 'REye')},
                       (0, 16): {'joint_names': ('Nose', 'LEye')},
                       (1, 0): {'joint_names': ('Neck', 'Nose')},
                       (1, 2): {'joint_names': ('Neck', 'RShoulder')},
                       (1, 5): {'joint_names': ('Neck', 'LShoulder')},
                       (1, 8): {'joint_names': ('Neck', 'MidHip')},
                       (2, 3): {'joint_names': ('RShoulder', 'RElbow')},
                       (2, 17): {'joint_names': ('RShoulder', 'REar')},
                       (3, 4): {'joint_names': ('RElbow', 'RWrist')},
                       (5, 6): {'joint_names': ('LShoulder', 'LElbow')},
                       (5, 18): {'joint_names': ('LShoulder', 'LEar')},
                       (6, 7): {'joint_names': ('LElbow', 'LWrist')},
                       (8, 9): {'joint_names': ('MidHip', 'RHip')},
                       (8, 12): {'joint_names': ('MidHip', 'LHip')},
                       (9, 10): {'joint_names': ('RHip', 'RKnee')},
                       (10, 11): {'joint_names': ('RKnee', 'RAnkle')},
                       (11, 22): {'joint_names': ('RAnkle', 'RBigToe')},
                       (11, 24): {'joint_names': ('RAnkle', 'RHeel')},
                       (12, 13): {'joint_names': ('LHip', 'LKnee')},
                       (13, 14): {'joint_names': ('LKnee', 'LAnkle')},
                       (14, 19): {'joint_names': ('LAnkle', 'LBigToe')},
                       (14, 21): {'joint_names': ('LAnkle', 'LHeel')},
                       (15, 17): {'joint_names': ('REye', 'REar')},
                       (16, 18): {'joint_names': ('LEye', 'LEar')},
                       (19, 20): {'joint_names': ('LBigToe', 'LSmallToe')},
                       (22, 23): {'joint_names': ('RBigToe', 'RSmallToe')}}

joints = {"Nose" : 0,
          "Neck" : 1,
          "RShoulder" : 2,
          "RElbow" : 3,
          "RWrist" : 4,
          "LShoulder" : 5,
          "LElbow" : 6,
          "LWrist" : 7,
          "MidHip" : 8,
          "RHip" : 9,
          "RKnee" : 10,
          "RAnkle" : 11,
          "LHip" : 12,
          "RKnee" : 13,
          "LAnkle" : 14,
          "REye" : 15,
          "LEye" : 16,
          "REar" : 17,
          "LEar" : 18,
          "LBigToe" : 19,
          "LSmallToe" : 20,
          "LHeel" : 21,
          "RBigToe" : 22,
          "RSmallToe" : 23,
          "Rheel" : 24}

classes = {0 : "head_and_shoulders_front", 1 : "full_front", 2 : "head_and_shoulders_back", 3 : "full_back", 4 : "other"}

# classifies a set of keypoints
def classify_keypoints(kp):
    if kp[joints["Nose"]][2] > 0.1 and (kp[joints["LEye"]][2] > 0.1 and kp[joints["REye"]][2] > 0.1) and (kp[joints["REar"]][2] > 0.1 or kp[joints["LEar"]][2] > 0.1):
        if kp[joints["Neck"]][2] > 0.1 and (kp[joints["RShoulder"]][2] > 0.1 or kp[joints["LShoulder"]][2] > 0.1):
            if int(kp[joints["MidHip"]][2] > 0.1) and (kp[joints["RHip"]][2] > 0.1 or kp[joints["LHip"]][2] > 0.1):
                return 1
            else:
                return 0
        else: 
            return 4
    elif kp[joints["Neck"]][2] > 0.1 or (kp[joints["RShoulder"]][2] > 0.1 or kp[joints["LShoulder"]][2] > 0.1) and not (kp[joints["REar"]][2] > 0.1 and kp[joints["LEar"]][2] > 0.1):
        if kp[joints["MidHip"]][2] > 0.1 and (kp[joints["RHip"]][2] > 0.1 or kp[joints["LHip"]][2] > 0.1):
            return 3
        else:
            return 2
    else:
        return 4

# classifies patches given json data path and pickle file
def classify_patches(json_pose_path, pickle_data_file):
    imgs_dict = {"head_and_shoulders_front": [], "full_front": [], 
                 "head_and_shoulders_back" : [], "full_back" : [], "other" : []}
    
    with open(pickle_data_file, 'rb') as f:
        imgs, bbs, idxs = pickle.load(f)
    
    n = len(imgs)
    
    for k in tqdm(range(n)):
        
        with open(json_pose_path + f'/{k}_keypoints.json', 'r') as file:
            data = json.load(file)
            
        img, bb, idx = imgs[k], bbs[k], idxs[k]
            
        if data['people'] == []:
            imgs_dict['other'].append(img)
            
        else:
            kp = np.array(data['people'][0]['pose_keypoints_2d']).reshape((25, 3))
            cls = classes[classify_keypoints(kp)]
            imgs_dict[cls].append(img)
            
    for cls in imgs_dict.keys():
        print(f'{cls} has {len(imgs_dict[cls])} entries')

    return imgs_dict
    


# adapted from https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/2109
def draw_pose(img, person_keypoints, bb=(0,0,0,0), draw_bb=False):
    """ Draw pose skeleton on input image.

    Image should be an RGB image represented as a ndarray shaped like (H, W, 3)

    person_keypoints should be ndarray shaped like (25, 3).
    The row index is the joint number per the BODY_25 OpenPose format.
    The first and second column indices are the (x, y) coordinates of the joint, respectively
    The third column is the detection confidence for that joint.
    """
    
    pose_coords = person_keypoints[:, 0:2]
    confidences = person_keypoints[:, 2]
    print(pose_coords, pose_coords.shape)
    
    for k in range(25):
        pose_coords[k][0] += bb[0]
        pose_coords[k][1] += bb[1]
        
    for j1, j2 in JOINT_PAIRS_MAP_ALL.keys():
        x1, y1 = list(map(lambda v: int(round(v)), pose_coords[j1]))
        x2, y2 = list(map(lambda v: int(round(v)), pose_coords[j2]))
        c1, c2 = confidences[j1], confidences[j2]
        if c1 > 0.0:
            cv2.circle(img, (x1, y1), radius=4, color=(0, 0, 255), thickness=-1)
        if c2 > 0.0:
            cv2.circle(img, (x2, y2), radius=4, color=(0, 0, 255), thickness=-1)
        if c1 > 0.0 and c2 > 0.0:
            cv2.line(img, (x1, y1), (x2, y2), color=(0, 255, 128), thickness=2)
            
    if draw_bb:
        cv2.line(img, (bb[0], bb[1]), (bb[2], bb[1]), color=(0, 0, 255), thickness=2)
        cv2.line(img, (bb[0], bb[3]), (bb[2], bb[3]), color=(0, 0, 255), thickness=2)
        cv2.line(img, (bb[0], bb[1]), (bb[0], bb[3]), color=(0, 0, 255), thickness=2)
        cv2.line(img, (bb[2], bb[1]), (bb[2], bb[3]), color=(0, 0, 255), thickness=2)
    return img

# save files to folders
def save_classification_dictionaries_to_file(dict_list, destination_folder):
    keys = dict_list[0].keys()
    for key in keys:
        i = 0
        print(f"writing {key} patches to {destination_folder}/{key}")
        for d in dict_list:
            print(f"writing {len(d[key])} patches")
            for img in d[key]:
                file_path = os.path.join(destination_folder, f'{key}/{i}.jpg')
                cv2.imwrite(file_path, img)
                i += 1
        print(f'{key} has total {i} classifications')
    return 0                