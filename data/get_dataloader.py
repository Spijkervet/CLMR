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
    

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False, # do not shuffle test set
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
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
