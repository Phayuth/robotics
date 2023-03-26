import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from robot.planar_rr import planar_rr
from planar_rr_kinematic_dataset import planar_rr_generate_dataset

# the result is obviously wrong, but this is just a foundation of research methodology, we can find another training model that work

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

def check_cuda():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


robot = planar_rr()

def train_arc():
    
    X, y = planar_rr_generate_dataset(robot)
    X = torch.from_numpy(X).to(check_cuda()).float() # ==>> X.shape:  (129600, 2)
    y = torch.from_numpy(y).to(check_cuda()).float() # ==>> y.shape:  (129600, 2)

    # Creat Model
    model = forwd_model().to(check_cuda())

    # Create Loss Function and Optimization
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Create Training Function
    def training(X, y, model, loss_func, opt):
        model.train()
        for index in range(len(X)):

            xx = X[index]

            # 1st prediction compt
            pred = model(xx)
            yy = y[index]
            
            # 2nd loss compt
            loss = loss_func(pred,yy)
            
            # 3rd Back prog
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f"index : {index}, loss = {loss.item()}")
            

    # start training
    for epoch in range(1):
        training(X, y, model, criterion, optimizer)
        
    # Save model
    # torch.save(model, "./kinedyna_learn/forward_kinematic_nn.pth")


def eval_arc():
    model_load = torch.load("./kinedyna_learn/forward_kinematic_nn.pth")    
    model_load.eval()

    with torch.no_grad():
        theta = torch.tensor([0.2,0.2],dtype=float).to(check_cuda())
        predt_ee_pose = model_load(theta)
        print("==>> predt: ", predt_ee_pose)

    robot.plot_arm(np.array(theta.cpu()).reshape(2,1), plt_basis=True, plt_show=True)

if __name__=="__main__":
    train_arc()
    # eval_arc()