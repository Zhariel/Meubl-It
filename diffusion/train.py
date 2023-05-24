import numpy as np
import os
from torchvision import transforms
from PIL import Image

from diffusion.model.unet import UNet
from diffusion.model_original.unet import SimpleUnet
from diffusion.scheduler import *

from torch.optim import Adam

IMAGE_PATH = os.path.join('assets', 'image.jpg')
IMG_SIZE = 64
BATCH_SIZE = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

input_image = Image.open(IMAGE_PATH).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
input_tensor = preprocess(input_image).unsqueeze(0)

# model = UNet()
model = SimpleUnet()

model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100

t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
loss = get_loss(model, input_tensor, t, device)

# img = sample_timestep(input_tensor, t, model, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas,
#                       posterior_variance)

# for epoch in range(epochs):
#     for step, batch in enumerate(dataloader):
#       optimizer.zero_grad()
#
#       t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
#       loss = get_loss(model, batch[0], t)
#       loss.backward()
#       optimizer.step()
#
#       if epoch % 5 == 0 and step == 0:
#         print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
#         sample_plot_image()
#
# t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
# output_tensor = model(input_tensor, t)
# output_tensor = output_tensor.squeeze(0)
# # output_tensor = output_tensor.permute(1, 2, 0)
# output_tensor = output_tensor.detach().cpu().numpy()
#
# output_tensor = (output_tensor * 255).astype(np.uint8)
#
# output_image = Image.fromarray(output_tensor)
# output_image.save(IMAGE_PATH)
#
# output_image.show()
