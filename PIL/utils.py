import numpy as np
from PIL import Image

# example: timg(batch['obs']['left_wrist_img'][0][0]).save('hello.jpg')
# torch.Tensor
def timg(img):
    return Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

