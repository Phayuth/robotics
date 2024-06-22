import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from spatial_geometry.taskmap_geo_format import task_rectangle_obs_1
from robot.planar_rr import PlanarRR
from spatial_geometry.spatial_shape import ShapeLine2D, ShapeCollision


def collision_dataset(robot, obs_list):
    # start sample
    sample_size = 360
    theta_candidate = np.linspace(-np.pi, np.pi, sample_size)

    sample_theta = []  # X
    sample_collision = []  # y

    for th1 in range(sample_size):
        for th2 in range(sample_size):
            theta = np.array([[theta_candidate[th1]], [theta_candidate[th2]]])
            link_pose = robot.forward_kinematic(theta, return_link_pos=True)
            linearm1 = ShapeLine2D(link_pose[0][0], link_pose[0][1], link_pose[1][0], link_pose[1][1])
            linearm2 = ShapeLine2D(link_pose[1][0], link_pose[1][1], link_pose[2][0], link_pose[2][1])

            col = []
            for i in obs_list:
                col1 = ShapeCollision.intersect_line_v_rectangle(linearm1, i)
                col2 = ShapeCollision.intersect_line_v_rectangle(linearm2, i)
                col.extend((col1, col2))

            if True in col:
                sample_collision.append(1.0)
            else:
                sample_collision.append(0.0)

            sample_theta_row = [theta_candidate[th1], theta_candidate[th2]]
            sample_theta.append(sample_theta_row)

    return np.array(sample_theta), np.array(sample_collision)


def check_cuda():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class CustomDataset(Dataset):

    def __init__(self, datainput, datalabel):
        self.data = datainput
        self.label = datalabel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        y = torch.Tensor([self.label[idx]]).type(torch.float64)
        return (x, y)


robot = PlanarRR()
obs_list = task_rectangle_obs_1()
X, y = collision_dataset(robot, obs_list)

model = Net().type(torch.float64)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()


# Define the training loop
def train(model, optimizer, criterion, train_loader, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print statistics
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")


dataset = CustomDataset(X, y)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
num_epochs = 100
hist = train(model, optimizer, criterion, train_loader, num_epochs)

with torch.no_grad():
    theta = torch.tensor([0.0, 0.0]).type(torch.float64)
    collision = model(theta)
    print(f"==>> collision: \n{collision}")

# def configuration_generate_plannar_rr_neuralnet():
#     grid_size = 360
#     theta1 = np.linspace(-np.pi, np.pi, grid_size)
#     theta2 = np.linspace(-np.pi, np.pi, grid_size)

#     grid_map = np.zeros((grid_size, grid_size))

#     for th1 in range(len(theta1)):
#         for th2 in range(len(theta2)):
#             theta = torch.tensor([th1, th2]).type(torch.float64)
#             collision = model(theta)
#             grid_map[th2, th1] = collision

#     return 1 - grid_map

# import matplotlib.pyplot as plt

# map = configuration_generate_plannar_rr_neuralnet()
# plt.imshow(map)
# plt.show()