import torch

# z = w*x + b   -> prediction
# y             -> label
# error = z - y -> calculate error
# loss(error)   -> calculate loss in function of error
# we want to train (w) and (b)

x = torch.ones(5)  # input tensor
print("x",x)
y = torch.zeros(3)  # expected output
print("y",y)
w = torch.randn(5, 3, requires_grad=True) # weight
print("w",w)
b = torch.randn(3, requires_grad=True) # biased
print("b",b)
z = torch.matmul(x, w)+b
print("z",z)
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print("loss",loss)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Compute Gradient with derivative
loss.backward()
print(w.grad)
print(b.grad)

# Disable Gradient Tracking when we use eval, so it freeze the parameter and cost efficient
# Example with grad
z = torch.matmul(x, w)+b
print(z.requires_grad)
# Example with no grad
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)





# https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html