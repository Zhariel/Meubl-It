import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def a(tensor):
    tensor = tensor.squeeze()

    # Normalize the tensor values to the range [0, 1]
    # tensor = tensor.clamp(0, 1)
    transform = transforms.ToPILImage()
    image = transform(tensor)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Display the image
    image.show()


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def mask_image_tensor(t1, t2, x1, y1, x2, y2):
    # Merges the boxed in t2 defined by the coordinates onto t1
    # Therefore t1 should be the original image and t2 the transformed image
    # Tensor(1, 3, X, X)
    mask = torch.zeros_like(t1)

    mask[:, :, y1 - 1:y2, x1 - 1:x2] = 1
    masked_t2 = t2 * mask
    t1[:, :, y1 - 1:y2, x1 - 1:x2] = 0

    return t1 + masked_t2


def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, xy1, xy2, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )

    noisy_image = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
                  + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

    masked_noisy_image = mask_image_tensor(x_0, noisy_image, *xy1, *xy2)
    masked_noise = mask_image_tensor(torch.zeros_like(x_0), noise, *xy1, *xy2)
    a(masked_noisy_image)
    a(masked_noise)

    return masked_noisy_image.to(device), masked_noise.to(device)


@torch.no_grad()
def sample_timestep(x, t, model, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def get_loss(model, x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, box, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                                              (box[0], box[1]), (box[2], box[3]), device)
    # x_noisy, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

