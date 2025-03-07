import torch
from check_cuda import check_cuda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset_import import dataset_test

# Load Model
model = torch.load("./weight/modelsave.pt")
model.eval()

# Load Test Data
dataloader_test = DataLoader(dataset = dataset_test , batch_size = 10000, shuffle = False)

# Availble Class
classes_list = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"]

# Predict
index = 22
for imgt, targt in dataloader_test:
    pred = model(imgt.to(check_cuda()))
    predicted = classes_list[pred[index].argmax().item()]
    actual    = classes_list[targt[index].item()]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
    break

plt.imshow(imgt[index].squeeze())
plt.show()