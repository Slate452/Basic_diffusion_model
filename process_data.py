
import os 
from os import listdir
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BATCH_SIZE = 128
IMG_SIZE = 64

def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines


def get_data_files(img_dir ="./data/data",test_file = './data/Test.txt', train_file = './data/Train.txt', Trainingsplit = 0.7):
        #Check if training and test list of files exist 
        train_list , test_list =[], []
        img_list  = os.listdir(img_dir)
        if "Train.txt" in os.listdir('./data'): 
                """find better variable or question """
                test_list, train_list = read_file_to_list(test_file), read_file_to_list(train_file)
                return train_list, test_list 
        else:
                test_file, train_file = open(train_file, 'w'), open(test_file, 'w')
                for i, img in enumerate(img_list):
                        if i < int(len(img_list)*Trainingsplit):
                                train_file.write(f"{img}\n")
                                train_list.append(img)
                                
                        else:
                                test_file.write(f"{img}\n")
                                test_list.append(img)
                return train_list , test_list


# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, img_list, img_dir, transform=None):
        self.img_list = img_list
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_list[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

def show_tensor_image(img):
    r_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(img.shape) == 4:
        img = img[0, :, :, :] 
    plt.imshow(r_transforms(img))

# Function to get combined dataset
def get_and_load_dataset(img_dir = "./data/data"):
        # Define transformations
        f_transforms = transforms.Compose([
                        transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),  # Scales data into [0,1]
                        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
                         ])

        # Read training and testing lists
        training_list, testing_list = get_data_files()

        # Create datasets
        train_dataset = CustomImageDataset(training_list, img_dir, transform=f_transforms)
        test_dataset = CustomImageDataset(testing_list, img_dir, transform=f_transforms)

        # Combine datasets
        combined_dataset = ConcatDataset([train_dataset, test_dataset])
        loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        print(f"Total images in combined dataset: {len(combined_dataset)}")
        return combined_dataset, loader

