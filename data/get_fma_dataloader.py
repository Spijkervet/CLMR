import os
import glob
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from data.fma import FmaDataset
import mirdata

import sys
from modules.transformations import AudioTransforms


def get_fma_loaders(args, num_workers=16):

    train_dataset = FmaDataset(
        args,
        train=True,
        audio_length=args.audio_length,
        transform=AudioTransforms(args),
    )

    test_dataset = FmaDataset(
        args,
        train=False,
        audio_length=args.audio_length,
        transform=AudioTransforms(args),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_loader, train_dataset, test_loader, test_dataset
