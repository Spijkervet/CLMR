import os
import torch
from pathlib import Path
import numpy as np

from modules.transformations import TransformsSimCLR

# from modules.transformations import SIm 
from .deepscores import DeepScoresDataset

def get_deepscores_dataloader(args, num_workers=16):

    # train
    train_dataset = DeepScoresDataset( 
        args, train=True, transform=TransformsSimCLR() #AudioTransforms(args)
    )

    # test
    test_dataset = DeepScoresDataset( 
        args, train=False, transform=TransformsSimCLR() #AudioTransforms(args)
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
        shuffle=False,  # do not shuffle test set
        drop_last=True,
        num_workers=num_workers,
    )

    args.n_classes = args.num_tags
    return train_loader, train_dataset, test_loader, test_dataset
