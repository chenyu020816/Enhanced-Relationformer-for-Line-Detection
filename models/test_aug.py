import os
import random
import torch
import numpy as np
import cv2
import pyvista
from torchvision.utils import save_image
import yaml

import os
import numpy as np
import random
import json
import scipy.ndimage
import torch
import pyvista
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
import cv2

from dataset_road_network import build_road_network_data
from augmentations import *

# train_transform = Compose(
#     [
#         Flip,
#         Rotate90,
#         ToTensor,
#     ]
# )
# train_transform = None
def aug_pipeline(data):
    data = hori_flip(data)
    data = vert_flip(data)
    data = random_hide(data, max_hide_size=(30, 30), p=1)
    data = random_add_point(data, p=0.2)
    data = jpeg_compress(data)
    data = gaussian_blur(data)
    return data


train_transform = ComposeLineData([
    lambda x: aug_pipeline(x)
    
])
# val_transform = Compose(
#     [
#         ToTensor,
#     ]
# )
val_transform = None


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def save_image_with_lines(image, coordinates, lines, out_path):
    for line in lines:
        p1 = list(coordinates[line[0]] * image.shape[0])
        p2 = list(coordinates[line[1]] * image.shape[0])
        p1 = (int(p1[1]), int(p1[0]))
        p2 = (int(p2[1]), int(p2[0]))
        cv2.circle(image, p1, color=(255, 0, 0), radius=1)
        cv2.circle(image, p2, color=(255, 0, 0), radius=1)
        cv2.line(image, p1, p2, color=(0, 255, 0), thickness=1)

    cv2.imwrite(out_path, image[..., ::-1])  # RGB -> BGR


def test_dataset(config, save_dir="debug_output", num_samples=20):
    os.makedirs(save_dir, exist_ok=True)

    dataset = build_road_network_data(config, mode='train')  # or mode='split' / 'test'

    print(f"Total samples: {len(dataset)}")
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        print(idx)
        image, seg, coords, lines = dataset[idx]
        image = denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        
        image = (image * 1).astype(np.uint8).copy()
        img_path = os.path.join(save_dir, f"sample_{idx}_img.png")
        # cv2.imwrite(img_path, image[..., ::-1])

        vis_path = os.path.join(save_dir, f"sample_{idx}_vis.png")
        save_image_with_lines(image, coords.numpy(), lines.numpy(), vis_path)

        seg_path = os.path.join(save_dir, f"sample_{idx}_seg.png")
        seg_np = (seg.squeeze().numpy() * 255).astype(np.uint8)
        # cv2.imwrite(seg_path, seg_np)

        


def main():
    with open("./configs/test_config.yaml") as f:
        print('\n*** Config file')
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)
    print(config.DATA.DATA_PATH)
    test_dataset(config, save_dir="debug_vis", num_samples=100)


if __name__ == "__main__":
    main()
