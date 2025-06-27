#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import config
import sys
sys.path.append(config.root_path)
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
from util import show_dataset_prev

#jupyter nbconvert --to script foot_dataset.ipynb


# In[ ]:


class FootDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        self.mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask  = Image.open(self.mask_paths[idx]).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            if 1 in np.unique(mask):
                mask_np = np.array(mask) * 255
                mask = Image.fromarray(mask_np.astype(np.uint8))
            mask = self.mask_transform(mask)

        return image, mask

def get_foot_datasets(dataset_dir, resolution=224, batch_size=16, augmented=False):

    # Definindo as transformações para normalizar e converter para tensor
    image_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),  # Converte para tensor de 1 canal
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),             # mantém valores 0 ou 1 se imagem for binária
        transforms.Lambda(lambda x: (x > 0.5).float())  # binariza se tiver 255
    ])


    if augmented:
        aug = "_aug"
        train_path = f'{dataset_dir}train{aug}'
        if os.path.exists(train_path) == False:
            train_path = f'{dataset_dir}train'

        validation_path = f'{dataset_dir}validation{aug}'
        if os.path.exists(validation_path) == False:
            validation_path = f'{dataset_dir}validation'

        test_path = f'{dataset_dir}test{aug}'
        if os.path.exists(test_path) == False:
            test_path = f'{dataset_dir}test'
    else:
        train_path      = f'{dataset_dir}train'
        validation_path = f'{dataset_dir}validation'
        test_path       = f'{dataset_dir}test'


    train_dataset = FootDataset(
        image_dir=f'{train_path}/images',
        mask_dir=f'{train_path}/labels',
        image_transform=image_transform,
        mask_transform=mask_transform
    )

    test_dataset = FootDataset(
        image_dir=f'{test_path}/images',
        mask_dir=f'{test_path}/labels',
        image_transform=image_transform,
        mask_transform=mask_transform
    )

    val_dataset = FootDataset(
        image_dir=f'{validation_path}/images',
        mask_dir=f'{validation_path}/labels',
        image_transform=image_transform,
        mask_transform=mask_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)   
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return train_loader, test_loader, val_loader


if __name__ == '__main__':
    dataset_path = config.medetec_dataset_path
    #dataset_path = config.fuseg_dataset_path
    train_loader, test_loader, val_loader = get_foot_datasets(dataset_path, resolution=224, batch_size=16, augmented=True)
    print(len(train_loader.dataset), len(test_loader.dataset), len(val_loader.dataset), 'total:', len(train_loader.dataset)+ len(test_loader.dataset)+ len(val_loader.dataset))

    show_dataset_prev(train_loader, test_loader, val_loader, num_images=3)


# In[ ]:




