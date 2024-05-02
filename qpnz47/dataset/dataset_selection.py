############################################################
# this file has all the necessary functions for question 1.3
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
from dataset_utils import *

# preprocesses the image so that it is compatible with ResNet
def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# extracts features from a directory full of images
# uses inception_v3 feature maps
def extract_features(folder_path):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg'))]
    imgs = [load_and_preprocess_image(path) for path in image_paths]
    feature_array = []
    
    # load ResNet
    model = models.inception_v3(pretrained=True)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for img in tqdm(imgs, desc='extracting features'):
            features = model(img.to(device)).squeeze(0)
            feature_array.append(features.cpu())
    return image_paths, feature_array

# returns file paths of images with similar features
def find_similar_images(image_paths, features_list, threshold=0.9):
    
    # compute pairwise cosine similarities
    similar_pairs = []
    for i in tqdm(range(len(features_list)), desc='comparing features'):
        for j in range(i + 1, len(features_list)):
            sim = cosine_similarity(features_list[i].unsqueeze(0), features_list[j].unsqueeze(0)).item()
            if sim > threshold:
                similar_pairs.append((image_paths[i], image_paths[j], sim))
    
    return similar_pairs

# returns an array of file paths to omit from training dataset
def img_files_to_remove(similar_img_files, thresh=0):
    to_remove = []
    for a, b, c in tqdm(similar_img_files):
        if c > thresh:
            if a in to_remove or b in to_remove:
                pass
            else:
                to_remove.append(a)
    return to_remove

# based on https://github.com/uoguelph-mlrg/instance_selection_for_gans/tree/master?tab=readme-ov-file
# https://arxiv.org/abs/2007.15255
def GaussianModel(embeddings):
    gmm = GaussianMixture(n_components=1, reg_covar=1e-05)
    gmm.fit(embeddings)

    log_likelihood = gmm.score_samples(embeddings)
    return log_likelihood

# based on https://github.com/uoguelph-mlrg/instance_selection_for_gans/tree/master?tab=readme-ov-file
# https://arxiv.org/abs/2007.15255
def get_keep_indices(embeddings, 
                     retention_ratio):
    
    keep_indices = []
    embeddings = torch.stack(embeddings)
    
    # compute the scores using the gaussian mixture model
    scores = GaussianModel(embeddings)
    
    # get the cutoff for the scores
    cutoff = np.percentile(scores, (100 - retention_ratio))
    
    # decide which indices to keep
    keep_indices = torch.from_numpy(scores > cutoff).bool()
    
    return keep_indices

def write_omit_file_images(folder_path, threshold=0.95, retention_ratio=0.8):
    
    # first compute similar image pairs and remove such pairs
    image_paths, embeddings = extract_features(folder_path)
    similar_image_pairs = find_similar_images(image_paths, embeddings, threshold)
    to_remove_similar_files = sorted(img_files_to_remove(similar_image_pairs))
    
    # now apply the gaussian instance selection
    reduced_img_paths = [image_paths[k] for k in range(len(image_paths)) if image_paths[k] not in to_remove_similar_files]
    reduced_embeddings = [embeddings[k] for k in range(len(image_paths)) if image_paths[k] not in to_remove_similar_files]
    selected_indices =  get_keep_indices(reduced_embeddings, retention_ratio)
    unselected_files = [reduced_img_paths[k] for k in range(len(reduced_img_paths)) if selected_indices[k] == 0]
    
    
    
    print(f"removing {len(to_remove_similar_files)} files from dataset for being too similar, and a further {len(unselected_files)} for being unselected")
    
    to_remove = to_remove_similar_files + unselected_files
    
    with open(folder_path + '/omit.txt', 'w') as file:
        for x in tqdm(to_remove, desc='writing omit file'):
            file.write(f"{x}\n")
            
    return to_remove_similar_files, unselected_files, similar_image_pairs

   