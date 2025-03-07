import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(0.1306,0.3081)])
dataset_train = torchvision.datasets.MNIST(root="./",transform = transform,train = True,download=True) #download=True
dataset_test  = torchvision.datasets.MNIST(root="./",transform = transform,train = False,download=True)