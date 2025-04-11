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

class Sat2GraphDataLoader(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, data, transform):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.transform = transform

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        data = self.data[idx]
        image_data = cv2.imread(data['img'])[:, :, ::-1]  # BGR to RGB
        # cv2.imwrite(os.path.join("debug_vis", f"sample_{idx}_ori.png"), cv2.imread(data['img']))
        seg_data = cv2.imread(data['seg'])[:, :, ::-1]
        seg_data = torch.from_numpy(seg_data.copy()).long().unsqueeze(0)
        polydata = pyvista.read(data['vtp'])
        if len(self.transform) != 0:
            h, w, _ = image_data.shape
            graph = Graph(polydata, h, w)
            line_data = LineData(image_data, graph)
            new_line_data = self.transform(line_data)
            image_data = new_line_data.image
            polydata = new_line_data.graph.to_polydata()
        image_data = torch.from_numpy(image_data.copy()).permute(2, 0, 1).float() #/ 255.0
        image_data = tvf.normalize(image_data, mean=self.mean, std=self.std)
        coordinates = torch.from_numpy(np.asarray(polydata.points, dtype=np.float32))
        lines = torch.from_numpy(polydata.lines.reshape(-1, 3).astype(np.int64))
        # correction of shift in the data
        # shift = [np.shape(image_data)[0]/2 -1.8, np.shape(image_data)[1]/2 + 8.3, 4.0]
        # coordinates = np.float32(np.asarray(vtk_data.points))
        # lines = np.asarray(vtk_data.lines.reshape(-1, 3))

        # coordinates = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
        # lines = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)

        return image_data, seg_data-0.5, coordinates[:,:2], lines[:,1:], new_line_data


def build_road_network_data(config, mode='train', split=0.95):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """    
    img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
    seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
    vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
    img_files = []
    vtk_files = []
    seg_files = []

    for file_ in os.listdir(img_folder):
        file_ = file_[:-8]
        img_files.append(os.path.join(img_folder, file_+'data.png'))
        vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
        seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

    data_dicts = [
        {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
    if mode=='train':
        ds = Sat2GraphDataLoader(
            data=data_dicts,
            transform=train_transform,
        )
        return ds
    elif mode=='test':
        img_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'seg')
        vtk_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
            seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
            ]
        ds = Sat2GraphDataLoader(
            data=data_dicts,
            transform=val_transform,
        )
        return ds
    elif mode=='split':
        img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
        vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
            seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
            ]
        random.seed(config.DATA.SEED)
        random.shuffle(data_dicts)
        train_split = int(split*len(data_dicts))
        train_files, val_files = data_dicts[:train_split], data_dicts[train_split:]
        train_ds = Sat2GraphDataLoader(
            data=train_files,
            transform=train_transform,
        )
        val_ds = Sat2GraphDataLoader(
            data=val_files,
            transform=val_transform,
        )
        return train_ds, val_ds

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
        image, seg, coords, lines, _ = dataset[idx]
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
    with open("../configs/test_config.yaml") as f:
        print('\n*** Config file')
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)
    print(config.DATA.DATA_PATH)
    test_dataset(config, save_dir="debug_vis", num_samples=100)


if __name__ == "__main__":
    main()