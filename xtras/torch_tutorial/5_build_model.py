import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x
    
model = model().to("cuda")
print(model)

X = torch.rand(1,28,28, device="cuda")
pred = model(X)
print(pred)

softmax = nn.Softmax(dim=1)
pred_prob = softmax(pred)
y_pred = pred_prob.argmax(1)
print(y_pred)

# Testing with fake data
input_img = torch.rand(3,28,28)
print(input_img)
print(input_img.size())


pred_img = model(input_img.to("cuda"))
print(pred_img.size())


# View Parameter in model
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")




# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html