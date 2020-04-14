import os
import glob
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from data.fma import FmaDataset
import mirdata

import sys
import sklearn.preprocessing

sys.path.append("../fma")
from fma.utils import load as load_fma

from modules.transformations import AudioTransforms


def get_fma_loaders(args, num_workers=16):

    tracks_file = os.path.join(args.data_input_dir, "fma_metadata", "tracks.csv")

    print(tracks_file)
    tracks = load_fma(tracks_file)

    train_df = tracks["set", "split"] == "training"
    val_df = tracks["set", "split"] == "validation"
    test_df = tracks["set", "split"] == "test"

    # subset
    small = tracks["set", "subset"] <= "small"

    train_df = tracks.loc[small & train_df | val_df]
    # val_df = tracks.loc[small & val_df]
    test_df = tracks.loc[small & test_df]

    y_train = train_df["track"]["genre_top"]
    y_test = test_df["track"]["genre_top"]

    # fill missing genres with unknown token
    y_train = y_train.cat.add_categories("<UNK>")
    y_test = y_test.cat.add_categories("<UNK>")
    y_train = y_train.fillna("<UNK>")
    y_test = y_test.fillna("<UNK>")

    enc = sklearn.preprocessing.LabelEncoder()
    y_train = enc.fit_transform(y_train)
    y_test = enc.transform(y_test)

    all_y = np.concatenate((y_train, y_test), axis=0)
    args.n_classes = len(np.unique(all_y))

    train_dataset = FmaDataset(
        args,
        os.path.join(args.data_input_dir, "fma_small",),
        train_df,
        labels=y_train,
        audio_length=args.audio_length,
        transform=AudioTransforms(args),
    )

    test_dataset = FmaDataset(
        args,
        os.path.join(args.data_input_dir, "fma_small",),
        test_df,
        labels=y_test,
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
