import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

 # ToTensor = convert img in form of PIL or nparrary to tensor and scale value to (0-1)
 # Lamda = user defined function
 # in the example, we create a tensor of 0 with the shape 1row, 10column. 
 # then change the value from 0 to 1 at the index where the label is belong to the class. ex label = 4 -> tensor[4] = 1




 # https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html