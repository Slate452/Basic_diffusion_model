import process_data as prep
from process_data import IMG_SIZE, BATCH_SIZE
import Diffuser as diff
from Diffuser import T
import unet, simple_unet
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.optim import Adam


data, data_loader = prep.get_and_load_dataset()
model = unet.AUnet() 
device =  "cpu"
model.to(device)       
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100 # Try more!

def test_unet():
    #Test unet 
    #inputs = torch.randn((2, 3, 512, 512))
    t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
    inputs = data[0].unsqueeze(0)
    model = simple_unet.build_unet()
    y = model(inputs)
    y = y.squeeze(0)
    #prep.show_tensor_image(y)


def test_attention() ->None:
    #Test Multihead Attention and Transfromer Implimentation
    x=1

def run_Diff_model() :
    for epoch in range(epochs):
        for step, batch in enumerate(data_loader):
            optimizer.zero_grad() 
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            
            loss = diff.get_loss(model, batch[0].unsqueeze(0), t)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                diff.sample_plot_image(model,device)

#run_Diff_model()
test_unet()
