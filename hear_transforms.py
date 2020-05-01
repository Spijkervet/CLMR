import os
import argparse
import torch
import torchaudio

from data import get_dataset
from utils import post_config_hook
from experiment import ex

tmp_dir = ".tmp"
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args.lin_eval = False
    args.model_name = "clmr"
    args.n_gpu = torch.cuda.device_count()
    args.batch_size = args.batch_size * args.n_gpu
    args.epochs = args.epochs * args.n_gpu

    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    (train_loader, train_dataset, test_loader, test_dataset) = get_dataset(args)

    for step, ((x_i, x_j), y, track_id) in enumerate(train_loader):
        for idx, (xb_i, xb_j, b_tid) in enumerate(zip(x_i, x_j, track_id)):
            xb_i = train_loader.dataset.denormalise_audio(xb_i)
            xb_j = train_loader.dataset.denormalise_audio(xb_j)
            torchaudio.save(f"{tmp_dir}/{step}-{idx}-xi-({b_tid}).wav", xb_i, sample_rate=args.sample_rate)
            torchaudio.save(f"{tmp_dir}/{step}-{idx}-xj-({b_tid}).wav", xb_j, sample_rate=args.sample_rate)
        break