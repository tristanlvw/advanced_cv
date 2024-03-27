import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

def display_img(img):
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.show()
    
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

JOINTS_MAP_ALL = {"Nose" : 0,
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

# from https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/2109
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


