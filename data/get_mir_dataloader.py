import os
import torch

from .mirdataset import MIRDataset
from modules.transformations import AudioTransforms

def get_mir_loaders(args, num_workers=16, diff_train_dataset=None):
    
    if args.lin_eval:
        is_unlabeled = False
    else:
        is_unlabeled = True
        
    train_dataset = MIRDataset(
        args,
        train=True,
        unlabeled=is_unlabeled, # the whole dataset, incl. test set(!!)
        audio_length=args.audio_length,
        transform=AudioTransforms(args)
    )

    test_dataset = MIRDataset(
        args,
        train=False,
        unlabeled=False,
        audio_length=args.audio_length,
        diff_train_dataset=diff_train_dataset,
        transform=AudioTransforms(args)
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

    return train_loader, train_dataset, test_loader, test_dataset
