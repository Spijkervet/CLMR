import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from clmr.datasets import get_dataset, SplitMusicDataset
from clmr.models import Identity, SampleCNN
from clmr.utils import (
    yaml_config_hook,
    load_encoder_checkpoint,
    extract_representations,
    save_representations,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLMR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(
            f"The checkpoint file {args.checkpoint_path} could not be found."
        )

    # ------------
    # dataloaders
    # ------------
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")

    train_dataset = SplitMusicDataset(train_dataset, max_audio_length=args.audio_length)
    valid_dataset = SplitMusicDataset(valid_dataset, max_audio_length=args.audio_length)
    test_dataset = SplitMusicDataset(test_dataset, max_audio_length=args.audio_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
    )

    # ------------
    # encoder
    # ------------
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=args.supervised,
        out_dim=train_dataset.n_classes,
    )
    encoder.to(args.device)
    encoder.eval()

    state_dict = load_encoder_checkpoint(args.checkpoint_path, train_dataset.n_classes)
    encoder.load_state_dict(state_dict)

    # replace the last fully connected layer with an identity function,
    # so that we can extract the representations.
    encoder.fc = Identity()

    # ------------
    # extract representations from our training dataset
    # ------------
    start_time = time.time()
    r, y = extract_representations(train_loader, encoder, device=args.device)
    save_representations("train.pt", r, y)

    print(
        f"Extracted representations of {(len(r) * args.audio_length) / args.sample_rate / 60 / 60} hours of music in {time.time()-start_time} seconds"
    )

    r, y = extract_representations(valid_loader, encoder, device=args.device)
    save_representations("valid.pt", r, y)

    r, y = extract_representations(test_loader, encoder, device=args.device)
    save_representations("test.pt", r, y)
