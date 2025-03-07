import torch.nn as nn

class Model(nn.Module):
    def __init__(self, img_dim, nm_class = 2):
        super(Model,self).__init__()
        self.nm_class = nm_class
        self.linear1 = nn.Linear(img_dim*img_dim,img_dim*img_dim*2)
        self.linear2 = nn.Linear(img_dim*img_dim*2,nm_class)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(img_dim*img_dim*2)

    def forward(self, x):
        # we have image (8,1,img_dim,img_dim), we want to (8,img_dim*img_dim)
        x = x.reshape(x.shape[0],-1)
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x