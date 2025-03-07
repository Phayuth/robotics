import torch.nn as nn

class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten  = nn.Flatten()
        self.linear_1 = nn.Linear(28*28,512)
        self.relu_1   = nn.ReLU()
        self.linear_2 = nn.Linear(512,1000)
        self.relu_2   = nn.ReLU()
        self.linear_3 = nn.Linear(1000,10)
    
    def forward(self,x):
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        x = self.relu_2(x)
        x = self.linear_3(x)
        return x

class Model2(nn.Module): # https://www.kaggle.com/code/pankajj/fashion-mnist-with-pytorch-93-accuracy/notebook
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out