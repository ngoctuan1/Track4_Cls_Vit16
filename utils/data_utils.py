import logging
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import pandas as pd

import torch
from torch.utils.data import Dataset

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.io import read_image
from PIL import Image
import numpy as np
from utils.augment import MSRCP


logger = logging.getLogger(__name__) 

class Track4_Dataset(Dataset):
  def __init__(self, df, img_path, transform = None, target_transform = None):
    self.df = df
    self.img_path = img_path
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, index):
    row = self.df.loc[index]
    img = cv2.imread(f'{self.img_path}/{row.image_path}')
    label = int(row.class_id) - 1
    if self.transform is not None:
      img = self.transform(img)

    # if self.target_transform is not None:
    #   label = self.target_transform(label)

    return img, label

def get_loader(args, test = False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    msrcp = MSRCP()
    transform_train = transforms.Compose([
        # transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        MSRCP(),
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    target_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        valset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        valset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "track4":
        df_train = pd.read_csv("/home/ai2023/Desktop/AICity_Track4_2023/Track4_Cls_Vit16/data/train.csv")
        df_val = pd.read_csv("/home/ai2023/Desktop/AICity_Track4_2023/Track4_Cls_Vit16/data/val.csv")

        trainset = Track4_Dataset(df_train, args.img_path, transform=transform_train, target_transform=target_transform)
        valset = Track4_Dataset(df_val, args.img_path, transform=transform_train, target_transform=target_transform)


    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    val_sampler = SequentialSampler(valset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                             sampler=val_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if valset is not None else None

    if test:
        df_test = pd.read_csv("/home/ai2023/Desktop/AICity_Track4_2023/Track4_Cls_Vit16/data/test.csv")
        testset = Track4_Dataset(df_test, args.img_path, transform=transform_train, target_transform=target_transform)
        test_sampler = SequentialSampler(testset)
        test_loader = DataLoader(testset,
                              sampler=test_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader

