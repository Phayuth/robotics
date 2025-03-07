import torch
import torchvision

from torch.utils.data import DataLoader # Dataloader
from torch.utils.data import Dataset # For custom dataset to inherent from
from torchvision import datasets # Sample Data
from torchvision.transforms import ToTensor, functional # Processing data
import matplotlib.pyplot as plt # plot
from torchvision.utils import save_image

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# View dataset before loading into the loader
sample_idx = torch.randint(len(training_data), size=(1,)).item() # random an interger from 0 to lenght of dataset
print(sample_idx)
img, label = training_data[sample_idx]
print('Get image dimension -> return channel, height, weight',functional.get_dimensions(img))
print('Get image size -> return height, weight',functional.get_image_size(img))

img = img.squeeze()
plt.imshow(img, cmap='gray')
plt.title(str(labels_map[label]))
plt.show()


# Custom dataset
class mydata(Dataset):
    def __init__(self,img_dir,label_dir,transform=None,target_transform=None):
        self.img_dir = img_dir # read all the available img in folder in to list -> ['path1','path2',....] usually use glob.glob(folder name)
        self.label_dir = label_dir # read all the name of the label of the image
        self.transform = transform # transformation class used to agument the dataset
        self.target_transform = target_transform # transformation class used to agument the label

    def __len__(self):
        return len(self.img_dir)
    
    def __getitem__(self, index): # this method, we only get 1 img, for multiple img is handle by Dataloader
        img_single = torchvision.io.read_image(self.img_dir[index]) # reading img in is all up to user, but at the end it must be in tensor form
        label_single = self.label_dir[index] # read label
        if self.transform:
            img_single = self.transform(img_single) # processing img. such as crop, rotate ...
        if self.target_transform: # processing label. 
            label_single = self.target_transform(label_single)
        return img_single, label_single



# Load data to the loader
loader = DataLoader(dataset=training_data, shuffle=True)

# View dataset after load to the loader
train_features, train_labels = next(iter(loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")







# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
