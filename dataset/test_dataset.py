import os
from itertools import product

import nibabel as nib
import SimpleITK as sitk
import numpy as np
import torch
from skimage.measure import regionprops
from torch.utils.data import DataLoader, Dataset
from .transforms import Compose, Window, MinMaxNorm

class TestDataset(Dataset):

    def __init__(self, args, image_path):
        image = nib.load(image_path)
        self.image_affine = image.affine
        self.image = image.get_fdata().astype(np.int16)
        self.crop_size = args.crop_size
        self.centers = self._get_centers()
        self.transforms =  Compose([
                Window(args.lower, args.upper),
                MinMaxNorm(args.lower, args.upper)
            ])

    def _get_centers(self):
        dim_coords = [list(range(0, dim, self.crop_size // 2))[1:-1]\
            + [dim - self.crop_size // 2] for dim in self.image.shape]
        centers = list(product(*dim_coords))

        return centers

    def __len__(self):
        return len(self.centers)

    def _crop_patch(self, idx):
        center_x, center_y, center_z = self.centers[idx]
        patch = self.image[
            center_x - self.crop_size // 2:center_x + self.crop_size // 2,
            center_y - self.crop_size // 2:center_y + self.crop_size // 2,
            center_z - self.crop_size // 2:center_z + self.crop_size // 2
        ]

        return patch

    def __getitem__(self, idx):
        image = self._crop_patch(idx)
        center = self.centers[idx]

        if self.transforms:
            image = self.transforms(image)

        image = torch.tensor(image[np.newaxis], dtype=torch.float)

        return image, center

    @staticmethod
    def _collate_fn(samples):
        images = torch.stack([x[0] for x in samples])
        centers = [x[1] for x in samples]

        return images, centers

    @staticmethod
    def get_dataloader(dataset, batch_size, num_workers=0):
        return DataLoader(dataset, batch_size, num_workers=num_workers,
            collate_fn=TestDataset._collate_fn)
    
