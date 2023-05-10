# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from util.dataloader_med import RetinaDataset, Augmentation, Node21, ChestX_ray14, Covidx, CheXpert, MIMIC
from .custom_transforms import GaussianBlur
import torch
from .augment import new_data_aug_generator

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_dataset_chest_xray(split, args):

    transform = Augmentation(normalize="MIMIC").get_augmentation("full_224", split)

    if args.dataset == 'chestxray':
        data_list = getattr(args, f'{split}_list')
        dataset = ChestX_ray14(args.data_path, data_list, augment=transform, num_class=14)

    elif args.dataset == 'covidx':
        print(args.dataset)
        dataset = Covidx(data_dir=args.data_path, phase=split, transform=transform)

    elif args.dataset == 'node21':
        dataset = Node21(data_dir=args.data_path, phase=split, transform=transform)

    elif args.dataset == 'chexpert':
        if split == 'train':
            mode = 'train'
        else:
            mode = 'valid'
        data_list = getattr(args, f'{split}_list')
        dataset = CheXpert(csv_path=data_list, image_root_path=args.data_path, use_upsampling=False,
                             use_frontal=True, mode=mode, class_index=-1, transform=transform)
    
    # TODO!!!!
    elif args.dataset == 'MIMIC':
        dataset = MIMIC(data_dir=args.data_path, phase=split, transform=transform)

    else:
        raise NotImplementedError
    print(dataset)

    return dataset

def build_transform(is_train, args):

    if args.norm_stats is not None:
        if args.norm_stats == 'imagenet':
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            raise NotImplementedError
    else:
        try:
            if args.dataset == 'chestxray' or args.dataset == 'covidx' or args.dataset == 'chexpert':
                mean = (0.5056, 0.5056, 0.5056)
                std = (0.252, 0.252, 0.252)
            elif args.dataset == 'imagenet':
                mean = IMAGENET_DEFAULT_MEAN
                std = IMAGENET_DEFAULT_STD
            elif args.dataset == 'retina':
                mean = (0.5056, 0.5056, 0.5056)
                std = (0.252, 0.252, 0.252)
        except:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD


    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

