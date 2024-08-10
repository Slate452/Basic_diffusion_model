#!/usr/bin/env python3
import os
import process_data as prep
from process_data import IMG_SIZE, BATCH_SIZE
import Diffuser as diff
from Diffuser import T
import unet
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.optim import Adam

save_path = './models/dif_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('./models', exist_ok=True)

data, data_loader, test_Dloader = prep.get_and_load_dataset()
model = unet.build_unet()

model.to(device)       
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100 # Try more!


def get_single_input():
    i = data[0].unsqueeze(0)
    i.to(device)
    t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
    print(i.shape)
    #prep.plot(i)
    return i, t



def time_embeding()-> None:
    img,t = get_single_input()
    enc = unet.PositionalEncoding(embedding_dim=256, max_len=1000)
    embeder = unet.embed_time(6)
    t_enc = enc(t)
    embeder(img,t_enc,r =True)
    


def test_attention() ->None:
    #Test Multihead Attention and Transfromer Implimentation
    Done=False


def Run_net()-> None:
    #Test unet 
    inputs, t = get_single_input()
    model = unet.build_unet()
    y = model(inputs,t)
    prep.plot(y)


@torch.no_grad()
def run_Diff_model() :
    for epoch in range(epochs):
        for step, batch in enumerate(data_loader):
            batch.requires_grad_()
            batch.to(device)
            optimizer.zero_grad() 
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = diff.get_loss(model, batch[0].unsqueeze(0), t,device=device)
            print(" Epoch ", epoch,"Loss: ", loss," Step: ", step)
            loss.backward()
            optimizer.step()
            
            if epoch > 1 and epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, save_path)
                print(f"Model saved to {save_path}")

        



def load_model(model_path=save_path, device=device):
    model = unet.build_unet().to(device)  # Rebuild the model and move it to the appropriate device
    optimizer = Adam(model.parameters(), lr=0.001)  # Recreate the optimizer

    # Load the state dictionaries
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model, optimizer

run_Diff_model()