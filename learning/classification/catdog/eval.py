# Force to use GPU
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import torch
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Import custom dataset class
from dataset_import import dataset_import

# Import plot
import matplotlib.pyplot as plt

# Create Dataset Class from custom image in folder for Testing
dataset_test = dataset_import("./dataset/test/")

# Load Data into Dataloader for Testing
dataloader_test = DataLoader(dataset=dataset_test, batch_size=16, shuffle=True)

# Device handling: make sure model and data are on the same device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the saved model onto the chosen device
# If the file contains a full model object (saved via torch.save(model, ...))
# use map_location so tensors are loaded onto `device` directly.
model = torch.load("./weight/modelsave.pt", map_location=device)
model.to(device)
model.eval()

test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in dataloader_test:
        # move both data and target to the same device as the model
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

print("Test Set:")
print("Average Loss:", test_loss)
print("Correct Prediction:", correct)
print("Number of Test Sample", len(dataloader_test.dataset))
print("Percentage of Correct:", 100 * correct / len(dataloader_test.dataset), "%")

# Availble Class
classes_list = ["Cat", "Dog"]

index = 10

for imgt, targt in dataloader_test:
    # move batch to device for prediction, but keep a CPU copy for plotting
    imgt_dev = imgt.to(device)
    pred = model(imgt_dev)
    predicted = classes_list[pred[index].argmax().item()]
    actual = classes_list[targt[index].item()]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
    # plot the CPU tensor (squeeze channel dimension)
    plt.imshow(imgt[index].squeeze().cpu(), cmap="gray")
    plt.show()
    break
