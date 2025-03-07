import torch
import torchvision.models as models

# Load model from sample
model = models.vgg16(pretrained=True)

# Save only weight model
torch.save(model.state_dict(), 'model_weights.pth')

# Load only weight
model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))

# Save both structure and weight 
torch.save(model, 'model.pth')

# Load both structure and weight
model = torch.load('model.pth')

# Call after load model
model.eval()




# https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
