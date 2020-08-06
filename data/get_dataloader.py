import os
import torch
from pathlib import Path
import numpy as np

from modules.transformations import AudioTransforms
from .audio import datasets


def get_audio_dataloader(args, pretrain=True, download=False):

    Dataset = datasets[args.dataset]

    if pretrain:
        transforms = AudioTransforms(args)
    else:
        transforms = None

    train_dataset = Dataset(args, split="train", pretrain=pretrain, download=download, transform=transforms)

    val_dataset = Dataset(args, split="valid", pretrain=pretrain, transform=transforms)

    test_dataset = Dataset(args, split="test", pretrain=pretrain, transform=transforms)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, shuffle=True
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=(val_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=(test_sampler is None), # do not shuffle test set
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler
    )

    args.n_classes = train_dataset.num_tags
    return (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    )
