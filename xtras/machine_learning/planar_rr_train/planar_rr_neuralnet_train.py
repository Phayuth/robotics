""" Forward Kinematic Model Train using Neural Network for Planar RR
- Model : Linear (100) -> Tanh() -> Linear (100) -> Linear (2)

The result is obviously wrong(not good is what I mean), but this is just a foundation of research methodology, we can find another training model that work

"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from robot.planar_rr import PlanarRR
from planar_rr_kinematic_dataset import planar_rr_generate_dataset


# forward dynamic model
class forwd_model(nn.Module):

    def __init__(self):
        super(forwd_model, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# check if cuda is availble to train on
def check_cuda():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


# dataset loader
class CustomDataset(Dataset):

    def __init__(self, datainput, datalabel):
        self.data = datainput
        self.label = datalabel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).to(check_cuda()).float()
        y = torch.from_numpy(self.label[idx]).to(check_cuda()).float()
        return (x, y)

robot = PlanarRR()

def train_arc():

    # create dataset and load data
    X, y = planar_rr_generate_dataset(robot)
    dataset = CustomDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Creat Model
    model = forwd_model().to(check_cuda())

    # Create Loss Function and Optimization
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Create Training Function

    def training(dataloader, model, loss_func, opt):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):

            # 1st-step prediction compute
            pred = model(X)

            # 2nd-step loss compute
            loss = loss_func(pred, y)

            # 3rd-step Back propagation
            opt.zero_grad()
            loss.backward()
            opt.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training(dataloader, model, criterion, optimizer)
    print("Done!")

    # Save model
    torch.save(model, "./kinedyna_learn/planar_rr_train/forward_kinematic_nn.pth")  # actually give ok result for a neural network approach


def eval_arc():
    model_load = torch.load("./kinedyna_learn/planar_rr_train/forward_kinematic_nn.pth")
    model_load.eval()

    with torch.no_grad():
        theta = torch.tensor([0.0, 0.0]).to(check_cuda())
        predt_ee_pose = model_load(theta)
        print("==>> predt: ", predt_ee_pose)

    robot.plot_arm(np.array(theta.cpu()).reshape(2, 1), plt_basis=True, plt_show=True)


if __name__ == "__main__":
    train_arc()
    # eval_arc()