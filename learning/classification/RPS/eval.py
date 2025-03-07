# Force to use GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

# Import torch
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Import custom dataset class
from dataset_import import dataset_import
from check_cuda import check_cuda

# Import plot
import matplotlib.pyplot as plt

# Create Dataset Class from custom image in folder for Testing
dataset_test  = dataset_import('./data/test/')

# Load Data into Dataloader for Testing
dataloader_test = DataLoader(dataset = dataset_test , batch_size = 16, shuffle = True)

model = torch.load("./torch_grayimg/weight/modelsave.pt")
model.eval()
# print(model)

test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in dataloader_test:
        data, target = data.to(check_cuda()), target.to(check_cuda())
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

print('Test Set:')
print('Average Loss:' , test_loss)
print('Correct Prediction:', correct)
print('Number of Test Sample', len(dataloader_test.dataset))
print('Percentage of Correct:', 100 * correct / len(dataloader_test.dataset),'%')

# # Availble Class
# classes_list = [
#     "Rock",
#     "Paper",
#     "Scissor"]

# index = 2
# for imgt, targt in dataloader_test:
#     pred = model(imgt.to(check_cuda()))
#     predicted = classes_list[pred[index].argmax().item()]
#     actual    = classes_list[targt[index].item()]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')
#     break

# plt.imshow(imgt[index])
# plt.show()