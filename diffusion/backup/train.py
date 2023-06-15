import os

from diffusion.dataset import custom_dataset
from torch.utils.data import DataLoader

from diffusion.model.unet import SimpleUnet
from diffusion.backup.scheduler import *

from torch.optim import Adam

IMAGE_PATH = os.path.join('../assets', 'image.jpg')
IMG_SIZE = 64
BATCH_SIZE = 128
device = "cpu" if torch.cuda.is_available() else "cpu"

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

# model = UNet()
model = SimpleUnet()

data = custom_dataset(IMG_SIZE=64)
l = len(data)
BATCH_SIZE = 128 if l > 128 else l
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 1

# t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
# loss = get_loss(model, input_tensor, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
# d = torchvision.datasets.StanfordCars(root=".", download=False)

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        image, annotations, box = batch

        t = torch.randint(299, 300, (BATCH_SIZE,), device=device).long()

        loss = get_loss(model, image, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, box, device)
        print(loss)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            # sample_plot_image(IMG_SIZE, T, device, model, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)



