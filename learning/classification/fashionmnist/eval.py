import torch
from torch.utils.data import DataLoader
from dataset_import import test_data

# Load Data for Test
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# Load saved Model
model = torch.load("./weight/fashion_model.pth")
model.eval()

# Availble Class
classes_list = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"]

# Predict
pic = 50
x = test_data[pic][0]
y = test_data[pic][1]
with torch.no_grad():
    pred      = model(x.to("cuda"))
    predicted = classes_list[pred[0].argmax(0)]
    actual    = classes_list[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')