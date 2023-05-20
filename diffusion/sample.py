import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

MODEL_PATH = os.path.join('assets', 'sample_model.pkl')
IMAGE_PATH = os.path.join('assets', 'image.jpg')

class Sample(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.acti = nn.ReLU()
        self.output = nn.Conv2d(32, 3, kernel_size=(3, 3), stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = self.acti(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

model = Sample()

input_image = Image.open(IMAGE_PATH).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
input_tensor = preprocess(input_image).unsqueeze(0)

# torch.save(model.state_dict(), MODEL_PATH)

loaded_model = Sample()
loaded_model.load_state_dict(torch.load(MODEL_PATH))
loaded_model.eval()

output_tensor = model(input_tensor)
output_tensor = output_tensor.squeeze(0)
output_tensor = output_tensor.permute(1, 2, 0)
output_tensor = output_tensor.detach().cpu().numpy()

output_tensor = (output_tensor * 255).astype(np.uint8)

output_image = Image.fromarray(output_tensor)
output_image.save(IMAGE_PATH)

output_image.show()