from dataset_import import test_dataset
from torch.utils.data import DataLoader
import torch
from check_cuda import check_cuda
import matplotlib.pyplot as plt

batch_size = 64
num_classes = 10

# Load Data for Test
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

# Load Model
model = torch.load('./weight/model.pt')
model.eval()

# Predict
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(check_cuda())
        labels = labels.to(check_cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))

for img, targ in test_loader:
    print(img.size())
    print(targ.size())
    img_single = img[0]
    print(img_single.size())
    plt.imshow(img_single.T)
    plt.show()
    break