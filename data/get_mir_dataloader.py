import os
import torch

from .mirdataset import MIRDataset
from modules.transformations import AudioTransforms


def get_mir_loaders(args, num_workers=16, diff_train_dataset=None):

    train_dataset = MIRDataset(
        args,
        os.path.join(args.data_input_dir, f"{args.dataset}_samples"),
        os.path.join(args.data_input_dir, f"{args.dataset}_labels/train_split.txt"),
        audio_length=args.audio_length,
    )

    test_dataset = MIRDataset(
        args,
        os.path.join(args.data_input_dir, f"{args.dataset}_samples"),
        os.path.join(args.data_input_dir, f"{args.dataset}_labels/test_split.txt"),
        audio_length=args.audio_length,
        diff_train_dataset=diff_train_dataset,
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
