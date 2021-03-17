import os
import argparse
import torch
import torchaudio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from sklearn import metrics

# Audio Augmentations
from torchaudio_augmentations import (
    RandomApply,
    Compose,
    ComposeMany,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)

# SimCLR
from simclr.modules.resnet import get_resnet

from callback import PlotSpectogramCallback
from data import ContrastiveDataset
from datasets import get_dataset
from modules.sample_cnn import SampleCNN
from modules.shortchunk_cnn import ShortChunkCNN_Res
from model import ContrastiveLearning, SupervisedBaseline
from utils import yaml_config_hook


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # ------------
    # data augmentations
    # ------------
    if args.supervised:
        train_transform = [RandomResizedCrop(n_samples=args.audio_length)]
        num_augmented_samples = 1
    else:
        train_transform = [
            RandomResizedCrop(n_samples=args.audio_length),
            RandomApply([PolarityInversion()], p=args.transforms_polarity),
            RandomApply([Noise()], p=args.transforms_noise),
            RandomApply([Gain()], p=args.transforms_gain),
            RandomApply([HighLowPass(sample_rate=args.sample_rate)], p=args.transforms_filters),
            RandomApply([Delay(sample_rate=args.sample_rate)], p=args.transforms_delay),
            RandomApply([PitchShift(
                n_samples=args.audio_length,
                sample_rate=args.sample_rate,
            )], p=args.transforms_pitch),
            RandomApply([Reverb(sample_rate=args.sample_rate)], p=args.transforms_reverb)
        ]
        num_augmented_samples = 2

    spec_transform = []
    if not args.time_domain:
        n_fft = 512
        f_min = 0.0
        f_max = 8000.0
        n_mels = 128
        stype = "power"  # magnitude
        top_db = None  # f_max

        spec_transform = [
            torchaudio.transforms.MelSpectrogram(
                sample_rate=args.sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
            ),
            torchaudio.transforms.AmplitudeToDB(stype=stype, top_db=top_db),
        ]
        train_transform.extend(spec_transform)

    # ------------
    # dataloaders
    # ------------
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")
    contrastive_train_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=(1, args.audio_length),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
        ),
    )

    contrastive_valid_dataset = ContrastiveDataset(
        valid_dataset,
        input_shape=(1, args.audio_length),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
        ),
    )

    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True
    )

    valid_loader = DataLoader(
        contrastive_valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=False
    )

    # ------------
    # encoder
    # ------------
    if args.time_domain:
        encoder = SampleCNN(
            strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
            supervised=args.supervised,
            out_dim=train_dataset.n_classes,
        )
    else:
        encoder = ShortChunkCNN_Res(n_channels=128, n_classes=train_dataset.n_classes)
        # encoder = get_resnet(args.resnet)
        # encoder.conv1 = torch.nn.Conv2d(
        #     1, 64, kernel_size=7, stride=2, padding=3, bias=False
        # )

    # ------------
    # model
    # ------------
    args.accelerator = "dp"
    if args.supervised:
        l = SupervisedBaseline(args, encoder, output_dim=train_dataset.n_classes)
    else:
        l = ContrastiveLearning(args, encoder)

    logger = TensorBoardLogger("runs", name="CLMRv2-{}".format(args.dataset))
    if args.checkpoint_path:
        l = l.load_from_checkpoint(
            args.checkpoint_path, encoder=encoder, output_dim=train_dataset.n_classes
        )

    else:
        # ------------
        # training
        # ------------

        if args.supervised:
            early_stopping = EarlyStopping(monitor='Valid/loss', patience=20)
        else:
            early_stopping = None

        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[PlotSpectogramCallback()],
            logger=logger,
            sync_batchnorm=True,
            max_epochs=args.epochs,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            accelerator=args.accelerator
        )
        trainer.fit(l, train_loader, valid_loader)


    if args.supervised:
        if len(spec_transform):
            transform = Compose(spec_transform)
        else:
            transform = None

        contrastive_test_dataset = ContrastiveDataset(
            test_dataset,
            input_shape=(1, args.audio_length),
            transform=Compose(spec_transform),
        )

        est_array = []
        gt_array = []
        l = l.to("cuda:0")
        l.eval()
        with torch.no_grad():
            for idx in tqdm(range(len(contrastive_test_dataset))):
                _, label = contrastive_test_dataset[idx]
                batch = contrastive_test_dataset.concat_clip(idx, args.audio_length)
                batch = batch.to("cuda:0")
                output = l.encoder(batch)
                output = torch.nn.functional.softmax(output)
                output = output.mean(dim=0).argmax().item()
                est_array.append(output)
                gt_array.append(label)


        if args.dataset in ["magnatagatune"]:
            est_array = torch.stack(est_array, dim=0).cpu().numpy()
            gt_array = torch.stack(gt_array, dim=0).cpu().numpy()
            roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
            pr_aucs = metrics.average_precision_score(gt_array, est_array, average="macro")
            print("ROC-AUC:", roc_aucs)
            print("PR-AUC:", pr_aucs)

        # l.logger.experiment.add_scalar("Test/roc_auc", roc_aucs, args.epochs)
        # l.logger.experiment.add_scalar("Test/pr_auc", pr_aucs, args.epochs)

        accuracy = metrics.accuracy_score(gt_array, est_array)
        print("ACCURACY:", accuracy)

        # precision = metrics.precision_score(gt_array, est_array)
        # print("Precision:", precision)
        