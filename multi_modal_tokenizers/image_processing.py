import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import PIL
from dall_e import map_pixels

def preprocess(img, target_image_size):
    s = min(img.size)
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)
