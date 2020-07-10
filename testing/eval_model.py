import torch
import argparse
import os
import numpy as np

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset
from experiment import ex
from model import load_model

from utils import post_config_hook
from utils.eval import eval_all
from utils.youtube import download_yt
from datasets.utils.resample import convert_samplerate

import matplotlib.pyplot as plt

@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args.lin_eval = True

    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.batch_size = args.logistic_batch_size

    (train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset) = get_dataset(args)

    context_model, _, _ = load_model(args, reload_model=True, name=args.model_name)
    context_model.eval()

    args.n_features = context_model.n_features

    model, _, _ = load_model(args, reload_model=True, name="supervised")
    model = model.to(args.device)

    print(context_model)
    print(model)

    # initialize TensorBoard
    writer = SummaryWriter(log_dir=args.tb_dir)

    args.current_epoch = 0

    # eval all
    metrics = eval_all(
        args,
        test_loader,
        context_model,
        model,
        writer,
        n_tracks=None,
    )
     
    for k, v in metrics.items():
        print(f"[Test]: {k}: {v}")
