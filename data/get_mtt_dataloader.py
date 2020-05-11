import os
import torch
from pathlib import Path
import numpy as np

from modules.transformations import AudioTransforms
from .magnatagatune import MTTDataset

def get_mtt_loaders(args, diff_train_dataset=None):

    train_dataset = MTTDataset(
        args, train=True, transform=AudioTransforms(args)
    )

    val_dataset = MTTDataset(
        args, train=False, validation=True, transform=AudioTransforms(args)
    )

    test_dataset = MTTDataset(
        args, train=False, transform=AudioTransforms(args) 
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False
    )


    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # do not shuffle test set
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False
    )

    args.n_classes = train_dataset.num_tags
    return train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset
