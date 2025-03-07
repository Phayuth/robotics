from model import ConvNeuralNet
from check_cuda import check_cuda
from dataset_import import train_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20

# Load Data for Train
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

# Create Model
model = ConvNeuralNet(num_classes).to(check_cuda())

# Creat Optimazation
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

# Train
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(check_cuda())
        labels = labels.to(check_cuda())

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

print("Done")

# Save model
torch.save(model, './weight/model.pt')