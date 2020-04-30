import os
import torch
import torchvision
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset
from model import load_model, save_model
from solvers import CLMR, Supervised
from utils import eval_all, post_config_hook
from validation import audio_latent_representations, vision_latent_representations

#### pass configuration
from experiment import ex


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args.lin_eval = False  # first, pre-train, after that, lin. evaluation
    
    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    (train_loader, train_dataset, test_loader, test_dataset) = get_dataset(args)

    model, optimizer, scheduler = load_model(
        args, reload_model=args.reload, name=args.model_name
    )
    print(model.summary())

    writer = SummaryWriter(log_dir=args.tb_dir)

    # save random init. model
    args.current_epoch = "random"
    save_model(args, model, optimizer)

    args.global_step = 0
    args.current_epoch = 0
    if args.model_name == "supervised":

        supervised = Supervised(args, model)
        supervised.solve(train_loader, test_loader, args.start_epoch, args.epochs)
        auc, ap = eval_all(
            args, test_loader, None, supervised.model, writer, n_tracks=None
        )
        print(f"Final: AUC: {auc}, AP: {ap}")

    elif args.model_name == "clmr":
        clmr = CLMR(args, model, optimizer, scheduler, writer)
        clmr.solve(args, train_loader, test_loader, args.start_epoch, args.epochs)
    else:
        raise NotImplementedError()

    ## end training
    save_model(args, model, optimizer, name=args.model_name)
