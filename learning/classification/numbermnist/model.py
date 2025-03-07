import torch.nn as nn

class Model(nn.Module):
    def __init__(self,nm_class = 10):
        super(Model,self).__init__()
        self.nm_class = nm_class
        self.linear1 = nn.Linear(28*28,28*28*2)
        self.linear2 = nn.Linear(28*28*2,10)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(28*28*2)

    def forward(self, x):
        # we have image (8,1,28,28), we want to (8,28*28)
        x = x.reshape(x.shape[0],-1)
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        