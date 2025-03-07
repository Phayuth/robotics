import torch

from torchvision import datasets # Sample Data
from torchvision.transforms import ToTensor # Processing data
from torchvision.utils import save_image

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

for i in range(10):
    img, label = test_data[i]
    save_image(img,'./'+str(labels_map[label]+str(i)),'png')