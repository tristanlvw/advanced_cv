########################################################################
# utils for part 2 of the coursework
########################################################################

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from pytorch_fid.fid_score import calculate_fid_given_paths
import cv2
from tqdm import tqdm
import torch

def check_and_brighten(image, threshold, brightness_factor):
    # Calculate the average brightness of the image and normalize
    avg_brightness = np.mean(image)
    avg_brightness /= 255.0

    # brighten image if it falls under threshold value
    if avg_brightness < threshold:
        brightened_image = np.clip(image * (1 + brightness_factor), 0, 255).astype(np.uint8)
        return brightened_image
    return image

# wrapper function for brightening transformation
def brighten_dim_images(image, **kwargs):
    factor = 0.4 + random.uniform(-0.2, 0.2)
    return check_and_brighten(image, threshold=0.5, brightness_factor=factor)

tf = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Lambda(image=brighten_dim_images, p=0.7),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ], additional_targets={"image0": "image"},
)

##### Dataset class for domain transfer
# based on cycleGAN practical https://github.com/ylongresearch/COMP4107-ACV/blob/main/CycleGAN_example.ipynb
class AB_Dataset(Dataset):
    def __init__(self, path_A, path_B, transform=tf):
        self.paths_A = path_A
        self.paths_B = path_B
        self.transform = tf

        self.A_images = []
        self.B_images = []
        
        self.A_images += [(path_A + '/' + k) for k in os.listdir(path_A) if k[-1] == 'g']
        self.B_images += [(path_B + '/' + k) for k in os.listdir(path_B) if k[-1] == 'g']

        
        self.length_dataset = max(len(self.A_images), len(self.B_images))
        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        if index > self.A_len:
            B_img_path = self.B_images[random.randint(0, self.A_len - 1)]

        B_img_path = self.B_images[index % self.B_len]   
        A_img_path = self.A_images[index % self.A_len]

        A_img = np.array(Image.open(A_img_path).convert("RGB"))
        B_img = np.array(Image.open(B_img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=A_img, image0=B_img)
            A_image = augmentations["image"]
            B_image = augmentations["image0"]

        return {"A": A_image, "B": B_image}

class multiclass_AB_Dataset(Dataset):
    def __init__(self, paths_dict_A, paths_dict_B, transform=tf):
        # a path dict looks like {0 : "path_to_head_shoulders_front", 1 : "path_to_full_body", 2 : "path_to_head_shoulders_back", 3 : "path_to_full_back", 
        #                         4 : "path_to_other"}
        
        self.paths_dict_A = paths_dict_A
        self.paths_dict_B = paths_dict_B
        self.transform = tf

        self.A_images = []
        self.B_images = []


        for cls in paths_dict_A.keys():
            omitted_A = read_txt(f'{paths_dict_A[cls]}/omit.txt')
            omitted_B = read_txt(f'{paths_dict_B[cls]}/omit.txt')
        
            for file in os.listdir(paths_dict_A[cls]):
                a = f'{paths_dict_A[cls]}/{file}'
                if a not in omitted_A and a[-1] == 'g':
                    self.A_images.append((a, cls))
                    
            for file in os.listdir(paths_dict_B[cls]):
                b = f'{paths_dict_B[cls]}/{file}'
                if b not in omitted_B and b[-1] == 'g':
                    self.B_images.append((b, cls))
        
        self.length_dataset = max(len(self.A_images), len(self.B_images))
        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        # A is the game domain which has more images. We pick a random B (movie) sample if the index surpasses length of B
        if index > self.B_len:
            B_img_path, B_class = self.B_images[random.randint(0, self.B_len - 1)]

        B_img_path, B_class = self.B_images[index % self.B_len]   
        A_img_path, A_class = self.A_images[index % self.A_len]

        A_img = np.array(Image.open(A_img_path).convert("RGB"))
        B_img = np.array(Image.open(B_img_path).convert("RGB"))
        
        # print(f'A shape: {A_img.shape}\n')
        # print(f'B shape: {B_img.shape}\n')

        if self.transform:
            augmentationsA = self.transform(image=A_img)
            A_image = augmentationsA["image"]
            augmentationsB = self.transform(image=B_img)
            B_image = augmentationsB["image"]

        # now, we return the patch class as well as the image
        return {"A": (A_image, torch.tensor(A_class)), "B": (B_image, torch.tensor(B_class))}

def read_txt(filepath):
    with open(filepath, 'r') as file:
        lines = [line.strip() for line in file]
    return lines


# Helper function to display an image
def show_image(img, figsize=(12, 12), norm=True):
    if norm:
        img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=figsize)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# passes the videos frames through the generator 
def process_video_through_generator(video_path, generator, device, output_file='processed.mp4'):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    generator.eval()
    
    # Define the transformation: resize to 256x256 and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    for k in tqdm(range(total_frames), desc='Processing frames'):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = to_pil_image(frame_rgb)     
        transformed_frame = transform(pil_image)
        
        input_tensor = transformed_frame.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = generator(input_tensor)
        frames.append(output*0.5+0.5)
        
        output_image = to_pil_image(output.squeeze(0).cpu())
        
    cap.release()
    tensors_to_video(frames, output_file)

# converts an array of tensors to video
def tensors_to_video(tensors, output_video_path, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = 256, 256 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for tensor in tqdm(tensors, desc='Saving frames'):
        image = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        out.write(image)

    out.release()
    print("Video saved to", output_video_path)

# measures FID score between the movie and game sets
def measure_FID_score(real_movie_dir, real_game_dir, fake_movie_dir, fake_game_dir):
    fid_game_to_movie = calculate_fid_given_paths([real_movie_dir, fake_movie_dir], batch_size=50, device='cuda', dims=2048)
    fid_movie_to_game = calculate_fid_given_paths([real_game_dir, fake_game_dir], batch_size=50, device='cuda', dims=2048)
    return {"game_to_movie" : fid_game_to_movie, "movie_to_game" : fid_movie_to_game}

# takes in the metrics dictionary defined in the dclgan file and 
def output_graphs(model_metrics, question_name, plot_CLS_loss=False):
    
    if plot_CLS_loss:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
        ax3.set_ylabel("Loss")
        ax3.set_xlabel("Epochs")
        ax3.set_title(f"{question_name}Multi-class Losses")
        
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) 

    ax1.set_ylabel("Output Value")
    ax1.set_xlabel("Epochs")
    ax1.set_title(f"{question_name}DCLGAN Discriminator Outputs")

    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_title(f"{question_name}DCLGAN Model Losses")

    x = [k for k in range(len(model_metrics['G_A']))]
    
    for metric in model_metrics.keys():
        
        if metric[-1] in ['A', 'B', '1', '2']:
            ax2.plot(x, model_metrics[metric], label=metric)

        elif metric[-1] in ['l', 'e']:
            ax1.plot(x, model_metrics[metric], label=metric)

        elif plot_CLS_loss and metric[-1] == 's':
            ax3.plot(x, model_metrics[metric], label=metric)
            
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    if plot_CLS_loss:
        ax3.legend(loc='upper right')
    # Display the plot
    plt.tight_layout()  # Adjust subplots to give some padding between them
    plt.show()

# gets nth frame from a video
def get_nth_frame(path, n):
    cap = cv2.VideoCapture(path)
    k = 0
    if cap.isOpened():
        for k in tqdm(range(n+1)):
            ret, frame = cap.read()
    return frame
