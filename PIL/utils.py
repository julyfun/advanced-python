import numpy as np
from PIL import Image

# torch.Tensor
def timg(img):
    return Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

