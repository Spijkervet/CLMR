import os
import argparse
import time
import torch
import numpy as np
from datetime import datetime

# TensorBoard
from torch.utils.tensorboard import SummaryWriter


from data import get_dataset
from model import load_model
from validation.audio.latent_representations import audio_latent_representations
from utils import post_config_hook

#### pass configuration
from experiment import ex

def validate_latent_space(args, model, writer):
    (train_loader, train_dataset, test_loader, test_dataset) = get_dataset(args)
    start_time = time.time()
    global_step = 0
    epoch = args.epoch_num
    step = 0
    audio_latent_representations(args, train_dataset, model, epoch, step, global_step, writer, train=True, max_tracks=None, vis=True)


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)

    # set start time
    args.time = time.ctime()

    # Device configuration
    args.device = torch.device("cpu") # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.current_epoch = args.start_epoch

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load model
    model, _, _ = load_model(args, reload_model=True)
    

    # initialize TensorBoard
    tb_dir = os.path.join(args.out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    try:
        validate_latent_space(args, model, writer)
    except KeyboardInterrupt:
        print("Interrupted")
