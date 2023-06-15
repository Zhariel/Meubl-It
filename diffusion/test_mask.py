from PIL import Image
import torch
import torchvision.transforms as transforms
import os

# Open the image
image = Image.open(os.path.join('assets', 'test2.png'))

x1 = 3
y1 = 3
x2 = 6
y2 = 6

def mask_image_tensor(fours, eights, x1, y1, x2, y2):
    # Merges the boxed in t2 defined by the coordinates onto t1
    # Therefore t1 should be the original image and t2 the transformed image
    # Tensor(1, 3, X, X)
    mask = torch.zeros_like(fours)

    mask[:, y1 - 1:y2, x1 - 1:x2] = 1
    masked_8 = eights * mask
    fours[:, y1 - 1:y2, x1 - 1:x2] = 0

    return fours + masked_8


# Resize to 8x8 pixels
resized_image = image.resize((8, 8))

# Convert to PyTorch tensor
tensor_transform = transforms.ToTensor()
tensor_image = tensor_transform(resized_image)

fours = torch.ones_like(tensor_image) * 4
eights = torch.ones_like(tensor_image) * 8
mask = torch.zeros_like(tensor_image)

# mask[:, y1 - 1:y2, x1 - 1:x2] = 1
# masked_8 = eights * mask
# fours[:, y1 - 1:y2, x1 - 1:x2] = 0
#
# final = fours + masked_8

#
final = mask_image_tensor(fours, eights, x1, y1, x2, y2)


# tensor_image[:, y1 - 1:y2, x1 - 1:x2] = 0

# print(tensor_image.shape)
print(final)
# print(tensor_image.shape)
# print(tensor_image)
# print(fours)


def get_index_from_list(a, b, c):
    return a


def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, x1, y1, x2, y2, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )

    box_mask = torch.zeros_like(x_0)
    tensor_image[:, y1 - 1:y2, x1 - 1:x2] = 0

    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
