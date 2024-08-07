
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
IMG_SIZE = 128

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


def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines

def get_data_files(img_dir ="./data/1_parameter/results",test_file = './data/1_parameter/test_cases.txt', train_file = './data/1_parameter/train_cases.txt', Trainingsplit = 0.7):
        #Check if training and test list of files exist 
        train_list , test_list =[], []
        img_list  = os.listdir(img_dir)
        if "train_cases.txt'" in os.listdir('./data'): 
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


class NumpyDatasetFromFileList(Dataset):
    def __init__(self,file_list,file_dir,transform = None ):
        self.file_list = file_list
        self.file_dir = file_dir
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.file_dir, self.file_list[idx])
        with np.load(file_path) as data:
            np_array = data['a']  # Extracting the array 'a' from the .npz file
        tensor = torch.from_numpy(np_array).float()  # Convert to PyTorch tensor
        return tensor


def get_and_load_dataset(img_dir = "./data/1_parameter/results"):
        # Define transformations
        f_transforms = transforms.Compose([
                        transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),  # Scales data into [0,1]
                        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
                         ])
        # Read training and testing lists
        train_file_list, test_file_list = get_data_files()
        # Create datasets
        train_dataset = NumpyDatasetFromFileList(train_file_list, file_dir=img_dir)
        test_dataset = NumpyDatasetFromFileList(test_file_list, file_dir=img_dir)

        combined_dataset = ConcatDataset([train_dataset, test_dataset])
        loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        print(f"Total images in combined dataset: {len(combined_dataset)}")
        return combined_dataset, loader

def move_batch_to_gpu(batch):
    if torch.cuda.is_available():
        batch = batch.cuda()
    return batch
def move_to_gpu(loader, data:any = 0):
    data.cuda()
    for batch in loader:
        batch = move_batch_to_gpu(batch)
        print(batch.size())

def plot_tensor_channels(tensor, cmap='viridis'):
    num_channels = tensor.shape[0]
    fig, axes = plt.subplots(1, num_channels, figsize=(15, 15))

    channel_names = ["vel","pressure", "so", "on", "so2", "on2"]
    
    if num_channels == 1:
        axes = [axes]  
    
    for i, ax in enumerate(axes):
        im = ax.imshow(tensor[i], cmap=cmap)
        ax.set_title(channel_names[i])
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    plt.show()



def plot(case:torch.tensor = torch.randn(6, 128, 128)):
    # Convert to NumPy array
    if case.ndim == 4 and case.size(0) > 1:
        print("The tensor is a batch of images with more than one element.")
        for s in range(0,5):
            d = case[s].squeeze(0)
            np_array = d.numpy()
            plot_tensor_channels(np_array, cmap='viridis')
    else:
        if case.ndim == 4:
            print("The tensor is a batch of images with one or fewer elements.")
            case = case[0].squeeze(0)
            np_array = case.numpy()
            plot_tensor_channels(np_array, cmap='viridis')
        else:
            np_array = case.numpy()
            plot_tensor_channels(np_array, cmap='viridis')
