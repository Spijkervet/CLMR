import essentia.standard
import os
import torch
import torchvision
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset
from model import load_model, save_model
from modules.sync_batchnorm import convert_model
from solvers import CLMR, Supervised, CPC
from utils import eval_all, post_config_hook, write_audio_tb, args_hparams
from validation import audio_latent_representations, vision_latent_representations

#### pass configuration
from experiment import ex


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args.lin_eval = False  # first, pre-train, after that, lin. evaluation
    args.n_gpu = torch.cuda.device_count()

    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    (train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset) = get_dataset(args)

    model, optimizer, scheduler = load_model(
        args, reload_model=args.reload, name=args.model_name
    )

    # weight init.
    if not args.reload:
        model.apply(model.initialize)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model = convert_model(model)
        model = model.to(args.device)
        print(model.module.summary())
    else:
        print(model.summary())

    writer = SummaryWriter(log_dir=args.tb_dir)

    # save random init. model

    if not args.reload:
        args.current_epoch = "random"
        args.train_stage = 0
        save_model(args, model, optimizer, args.model_name)

    args.global_step = 0
    args.current_epoch = 0

    # write a few audio files to TensorBoard for comparison
    write_audio_tb(args, train_loader, test_loader, writer)

    if args.model_name == "supervised":
        supervised = Supervised(args, model)
        supervised.solve(args, train_loader, test_loader, args.start_epoch, args.epochs)
        auc, ap = eval_all(
            args, test_loader, None, supervised.model, writer, n_tracks=None
        )
        print(f"Final: AUC: {auc}, AP: {ap}")
        writer.add_hparams(
            args_hparams(args), {"hparam/test_auc": auc, "hparam/test_ap": ap}
        )
    elif args.model_name == "clmr":
        clmr = CLMR(args, model, optimizer, scheduler, writer)
        # clmr.solve(args, train_loader, val_loader, test_loader, args.start_epoch, args.epochs)

        if args.supervised:
            test_loss_epoch, auc, ap = clmr.test_avg(args, test_loader)
            print(f"[Test]: ROC-AUC: {auc}, PR-AUC: {ap}")
            writer.add_hparams(args_hparams(args), {"hparam/test_loss": test_loss_epoch, "hparam/test_auc": auc, "hparam/test_ap": ap}) 
    elif args.model_name == "cpc":
        cpc = CPC(args, model, optimizer, scheduler, writer)
        cpc.solve(args, train_loader, test_loader, args.start_epoch, args.epochs)

        if args.supervised:
            test_loss_epoch, auc, ap = cpc.test_avg(args, test_loader)
            print(f"[Test]: ROC-AUC: {auc}, PR-AUC: {ap}")
            writer.add_hparams(args_hparams(args), {"hparam/test_loss": test_loss_epoch, "hparam/test_auc": auc, "hparam/test_ap": ap}) 
    else:
        raise NotImplementedError()

    ## end training
    save_model(args, model, optimizer, name=args.model_name)
