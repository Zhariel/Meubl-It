from PIL import Image
import numpy as np


def show_img(tensor, istensor=True, permute=True):
    a = tensor
    if istensor:
        tensor = tensor.squeeze(0)
        if permute:
            tensor = tensor.permute(1, 2, 0)
        tensor = tensor.detach().cpu().numpy()
    tensor = Image.fromarray(tensor.astype(np.uint8))
    tensor.show()