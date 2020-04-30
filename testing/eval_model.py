import torch
import argparse
import os

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset
from experiment import ex
from model import load_model

from utils import post_config_hook
from utils.eval import tagwise_auc_ap, eval_all, average_precision


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)
    args.lin_eval = True

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.batch_size = args.logistic_batch_size

    (train_loader, train_dataset, test_loader, test_dataset) = get_dataset(args)

    simclr_model, _, _ = load_model(args, reload_model=True, name="clmr")
    simclr_model.eval()

    args.n_features = simclr_model.n_features

    model, optimizer, _ = load_model(args, reload_model=True, name="supervised")
    model = model.to(args.device)

    criterion = torch.nn.BCEWithLogitsLoss()  # for tags

    # initialize TensorBoard
    tb_dir = os.path.join(args.out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    args.global_step = 0
    args.current_epoch = 0

    # eval all
    auc, ap = eval_all(
        args,
        test_loader,
        simclr_model,
        model,
        criterion,
        optimizer,
        writer,
        n_tracks=None,
    )

    print(f"[Test]: ROC-AUC {auc}")
    print(f"[Test]: PR-AUC {ap}")
