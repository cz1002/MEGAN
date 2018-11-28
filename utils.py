"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch.utils.data import DataLoader
from torchvision import transforms as T
import yaml
from torchvision.datasets import ImageFolder

def get_loader(image_dir, crop_size=178, image_size=128,
               batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())  # 数据随机水平翻转
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))  # 正则化
    transform = T.Compose(transform)
    dataset = ImageFolder(image_dir, transform)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=(mode == 'train'),
                             num_workers=num_workers)
    return data_loader


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)