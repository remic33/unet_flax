"""Use pytorch for data augmentation and loading because it was too long to implement in jax"""
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
import jax


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, size: float = (256,256),
                 mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.size = size
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(
                f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, size:tuple, is_mask):
        pil_img = pil_img.resize(size,
                                 resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        #if not is_mask:
        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]


            img_ndarray = img_ndarray / 255
        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(
            img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(
            mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.size, is_mask=False)
        mask = self.preprocess(mask, self.size, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')


def custom_collate_fn(batch):
    """Provides us with batches of numpy arrays and not PyTorch's tensors."""
    masks = []
    imgs = []
    for line in batch:
        imgs.append(line['image'].detach().numpy())
        masks.append(line['mask'].detach().numpy())


    masks = np.array(masks)
    imgs = np.array(imgs)

    return imgs, masks


def load_data(val_percent, images_dir, mask_dir, batch_size):
    dataset = BasicDataset(images_dir, mask_dir)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, shuffle=True, collate_fn=custom_collate_fn,
                              **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, collate_fn=custom_collate_fn,
                            drop_last=True, **loader_args)
    return train_loader, val_loader
