import torch
from torch import nn

from model import Model1
from torch.utils.data import DataLoader
from dataset_import import training_data
from check_cuda import check_cuda

# Load Train Data
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# Creat Model
model = Model1().to(check_cuda()) # for gpu
print(model)

# Create Loss Function and Optimization
loss_func = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(),lr=1e-3)

# Create Training Function
def train(dataloader, model, loss_func, opt):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(check_cuda()),y.to(check_cuda())
        
        # 1st prediction compt
        pred = model(X)
        
        # 2nd loss compt
        loss = loss_func(pred,y)
        
        # 3rd Back prog
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Start Training
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_func, opt)
print("Done!")

# Save model
torch.save(model, "./weight/fashion_model.pth")
print("Saved PyTorch Model State to model.pth")


# If we want to train and test at the same time
# # def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0

#     with torch.no_grad():
#         for X, y in dataloader:
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()

#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
