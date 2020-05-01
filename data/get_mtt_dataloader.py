import os
import torch
from pathlib import Path
import numpy as np

from modules.transformations import AudioTransforms
from .magnatagatune import MTTDataset

def get_mtt_loaders(args, diff_train_dataset=None):

    train_annotations = (
        Path(args.mtt_processed_annot) / "train_gt_mtt.tsv" # "train_50_tags_annotations_final.csv"
    )
    train_dataset = MTTDataset(
        args, annotations_file=train_annotations, train=True, transform=AudioTransforms(args)
    )

    test_annotations = (
        Path(args.mtt_processed_annot) / "test_gt_mtt.tsv" # "test_50_tags_annotations_final.csv"
    )
    test_dataset = MTTDataset(
        args, annotations_file=test_annotations, train=False, transform=AudioTransforms(args) 
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    args.n_classes = args.num_tags
    return train_loader, train_dataset, test_loader, test_dataset
