import os
import argparse
import torch
import torchaudio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from sklearn import metrics

from torchaudio_augmentations import Compose, RandomResizedCrop

# SimCLR
from simclr.modules.resnet import get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from callback import PlotSpectogramCallback
from datasets import get_dataset
from data import ContrastiveDataset
from modules.sample_cnn import SampleCNN
from modules.shortchunk_cnn import ShortChunkCNN_Res
from model import ContrastiveLearning, LinearEvaluation
from utils import yaml_config_hook, load_encoder_checkpoint


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.accelerator = None

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("That checkpoint does not exist")

    train_transform = [RandomResizedCrop(n_samples=args.audio_length)]
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
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")

    contrastive_train_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=(1, args.audio_length),
        transform=Compose(train_transform),
    )

    contrastive_test_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=(1, args.audio_length),
        transform=Compose(train_transform),
    )

    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True,
    )

    test_loader = DataLoader(
        contrastive_test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=False,
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

    n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer
    
    state_dict = load_encoder_checkpoint(args.checkpoint_path)
    encoder.load_state_dict(state_dict)

    cl = ContrastiveLearning(args, encoder)
    cl.eval()
    cl.freeze()

    l = LinearEvaluation(
        args,
        cl.encoder,
        hidden_dim=n_features,
        output_dim=train_dataset.n_classes,
    )

    if args.linear_checkpoint_path:
        l = l.load_from_checkpoint(
            args.linear_checkpoint_path,
            encoder=cl.encoder,
            hidden_dim=n_features,
            output_dim=train_dataset.n_classes,
        )
    else:
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[PlotSpectogramCallback()],
            logger=TensorBoardLogger(
                "runs", name="CLMRv2-eval-{}".format(args.dataset)
            ),
            sync_batchnorm=True,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            max_epochs=args.epochs,
        )
        trainer.fit(l, train_loader, test_loader)

    if len(spec_transform):
        transform = Compose(spec_transform)
    else:
        transform = None
        
    contrastive_test_dataset = ContrastiveDataset(
        test_dataset,
        input_shape=(1, args.audio_length),
        transform=transform,
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

            h0 = l.encoder(batch)
            output = l.model(h0)
            output = torch.nn.functional.softmax(output, dim=1)
            track_prediction = output.mean(dim=0)
            est_array.append(track_prediction)
            gt_array.append(label)

            # for l, tag in zip(label.reshape(-1), test_dataset.tags):
            #     if l:
            #         print("Ground truth:", tag)

            # for p, tag in zip(track_prediction, test_dataset.tags):
            #     if p:
            #         print("Predicted:", p, tag)
            # torchaudio.save("{}.wav".format(idx), batch.cpu().reshape(-1), sample_rate=22050)
            # exit()


    est_array = torch.stack(est_array, dim=0).cpu().numpy()
    gt_array = torch.stack(gt_array, dim=0).cpu().numpy()

    roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
    pr_aucs = metrics.average_precision_score(gt_array, est_array, average="macro")

    print("ROC-AUC:", roc_aucs)
    print("PR-AUC:", pr_aucs)
