from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from dataset_import import dataset_train
from model import Model
from check_cuda import check_cuda

# Load Dataset to Dataloader
dataloader_train = DataLoader(dataset=dataset_train, batch_size=8, shuffle=True)

# Creat Model with 10 Classes
model = Model(10).to(check_cuda())

# Create loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0003)

# Training
for epc in range(10):
    model.train()
    num_cor = 0
    num_samp = 0
    loss_list = []
    for data, target in dataloader_train:
        data = data.to(check_cuda())
        target = target.to(check_cuda())

        # forward
        output = model(data)
        loss = criterion(output, target)
        loss_list.append(loss)

        _, pred = output.max(1)

        num_cor += (pred == target).sum()
        num_samp += pred.size(0)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # grad decent or adam
        optimizer.step()
    acc = num_cor / num_samp

    print("Epoch : " + str(epc) + "   Acc:  " + str(acc))

print("Done")

# Save Model
torch.save(model, "./weight/modelsave.pt")
