import os
import glob
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from repositories.fma.utils import load as load_fma
from src.audio.data.fma import FmaDataset
import mirdata


def get_fma_loaders(options, num_workers=16):

    tracks_file = os.path.join(options.data_input_dir, "fma_metadata", "tracks.csv")
    tracks = load_fma(tracks_file)

    train_df = tracks["set", "split"] == "training"
    val_df = tracks["set", "split"] == "validation"
    test_df = tracks["set", "split"] == "test"

    # subset
    small = tracks["set", "subset"] <= "small"

    train_df = tracks.loc[small & train_df | val_df]
    # val_df = tracks.loc[small & val_df]
    test_df = tracks.loc[small & test_df]

    train_dataset = FmaDataset(
        options, os.path.join(options.data_input_dir, "fma_small",), train_df
    )

    test_dataset = FmaDataset(
        options, os.path.join(options.data_input_dir, "fma_small",), test_df
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=options.batch_size_multiGPU,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=options.batch_size_multiGPU,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_loader, train_dataset, test_loader, test_dataset
