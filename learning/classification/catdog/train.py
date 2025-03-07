# Force to use GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

# Import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

# Import custom dataset class and model
from dataset_import import dataset_import
from model import Model

# Create Dataset Class from custom image in folder for Training
dataset_train = dataset_import('./dataset/train/')

# Load Data into Dataloader for Training
dataloader_train = DataLoader(dataset = dataset_train , batch_size = 16, shuffle = True)

# View Model
img_dim = 50
model = Model(img_dim, nm_class = 2).to("cuda")
print(model)

# Creat Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(),lr = 0.0003)

# Train
for epc in range(100):
    model.train()
    num_cor = 0
    num_samp = 0
    loss_list = []
    for data,target in dataloader_train:
        data = data.to("cuda")
        target = target.to("cuda")

        #forward
        output = model(data)
        loss = criterion(output,target)
        loss_list.append(loss)
        
        _,pred = output.max(1)

        num_cor += (pred == target).sum()
        num_samp += pred.size(0)
        

        #backward
        optimizer.zero_grad()
        loss.backward()

        #grad decent or adam
        optimizer.step()
    acc = num_cor/num_samp
    print("Epoch : "+str(epc)+" | Acc: "+str(acc))

print("Done")

# Save Dataset
torch.save(model, "./weight/modelsave.pt")