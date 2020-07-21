import torch
import argparse
import os
import numpy as np

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset
from model import load_encoder

from utils import yaml_config_hook, parse_args
from utils.eval import eval_all
from utils.youtube import download_yt
from scripts.datasets.resample import resample

import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.batch_size = args.logistic_batch_size

    (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    ) = get_dataset(args, pretrain=False)

    # load pre-trained encoder
    encoder = load_encoder(args, reload=True)
    encoder.eval()
    encoder = encoder.to(args.device)

    model = torch.nn.Sequential(
        torch.nn.Linear(args.n_features, args.n_classes)
    )

    model.load_state_dict(
        torch.load(
            os.path.join(
                args.finetune_model_path,
                f"finetuner_checkpoint_{args.finetune_epoch_num}.pt",
            )
        )
    )
    model = model.to(args.device)

    # initialize TensorBoard
    writer = SummaryWriter()

    args.current_epoch = args.epoch_num

    # eval all
    metrics = eval_all(args, test_loader, encoder, model, writer, n_tracks=None,)
    print("### Final tag/clip ROC-AUC/PR-AUC scores ###")
    for k, v in metrics.items():
        if "hparams" in k:
            print("[Test average AUC/AP]:", k, v)
        else:
            for tag, val in zip(test_loader.dataset.tags, v):
                print(f"[Test {k}]\t\t{tag}\t{val}")
