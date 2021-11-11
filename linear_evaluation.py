import argparse
import os
import time
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from torchaudio_augmentations import Compose, RandomResizedCrop

from clmr.data import ContrastiveDataset
from clmr.datasets import get_dataset
from clmr.evaluation import evaluate
from clmr.models import SampleCNN
from clmr.modules import ContrastiveLearning, LinearEvaluation
from clmr.utils import (
    yaml_config_hook,
    load_encoder_checkpoint,
    load_finetuner_checkpoint,
)


def create_dataset_from_representations(fp: str):
    data = torch.load(fp)
    return TensorDataset(data["representation"], data["y"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLMR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    pl.seed_everything(args.seed)
    args.accelerator = None

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"The checkpoint file {args.checkpoint_path} could not be found.")

    # ------------
    # dataloaders
    # ------------
    if os.path.exists("train.pt"):
        train_dataset = create_dataset_from_representations("train.pt")
        valid_dataset = create_dataset_from_representations("valid.pt")
    else:
        transform = Compose([RandomResizedCrop(n_samples=args.audio_length)])
        train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
        valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")

        train_dataset = ContrastiveDataset(train_dataset, input_shape=[1, args.audio_length], transform=transform)
        valid_dataset = ContrastiveDataset(valid_dataset, input_shape=[1, args.audio_length], transform=transform)


    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")
    contrastive_test_dataset = ContrastiveDataset(test_dataset, input_shape=[1, args.audio_length], transform=None)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    # ------------
    # encoder
    # ------------
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=args.supervised,
        out_dim=test_dataset.n_classes,
    )

    n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer
    # assert n_features == train_dataset[0][0].shape[-1] # make sure dimensions of the extracted representations match.

    state_dict = load_encoder_checkpoint(args.checkpoint_path, test_dataset.n_classes)
    encoder.load_state_dict(state_dict)

    cl = ContrastiveLearning(args, encoder)
    # cl.eval()
    # cl.freeze()

    module = LinearEvaluation(
        args,
        cl.encoder,
        hidden_dim=n_features,
        output_dim=test_dataset.n_classes,
    )
    module.to(args.device)

    if args.finetuner_checkpoint_path:
        state_dict = load_finetuner_checkpoint(args.finetuner_checkpoint_path)
        module.model.load_state_dict(state_dict)
    else:
        early_stop_callback = EarlyStopping(
            monitor="Valid/loss", patience=15, verbose=False, mode="min"
        )

        trainer = Trainer.from_argparse_args(
            args,
            logger=TensorBoardLogger(
                "runs", name="CLMRv2-eval-{}".format(args.dataset)
            ),
            max_epochs=args.finetuner_max_epochs,
            callbacks=[early_stop_callback],
        )
        trainer.fit(module, train_loader, valid_loader)

    results = evaluate(
        module.encoder,
        module.model,
        contrastive_test_dataset,
        args.dataset,
        args.audio_length,
        device=args.device,
    )
    print(results)
