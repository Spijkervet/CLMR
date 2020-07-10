import essentia.standard
import os
import torch
import torchvision
import argparse
import numpy as np
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

from data import get_dataset
from model import load_optimizer, save_model
from modules.sync_batchnorm import convert_model
from modules import SimCLR, SampleCNN59049, NT_Xent
from solver import Solver
from utils import eval_all, yaml_config_hook, write_audio_tb, args_hparams
from validation import audio_latent_representations, vision_latent_representations

#### pass configuration


def main(gpu, args):

    # data loaders
    (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    ) = get_dataset(args)

    # encoder
    if args.domain == "audio":
        if args.sample_rate == 22050:
            encoder = SampleCNN59049(args)
        print(f"### {encoder.__class__.__name__} ###")
    elif args.domain == "scores":
        encoder = get_resnet(args.resnet, pretrained=False)  # resnet
        encoder.conv1 = nn.Conv2d(
            args.image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    else:
        raise NotImplementedError

    if args.reload:
        # reload model
        print(
            f"### RELOADING {args.model_name.upper()} MODEL FROM CHECKPOINT {args.epoch_num} ###"
        )
        model_fp = os.path.join(
            model_path, "{}_checkpoint_{}.tar".format(args.model_name, args.epoch_num)
        )
        encoder.load_state_dict(
            torch.load(model_fp, map_location=args.device.type), strict=True
        )

        optim_fp = os.path.join(
            model_path, "{}_checkpoint_{}_optim.tar".format(args.model_name, args.epoch_num)
        )
        print(f"### RELOADING {args.model_name.upper()} OPTIMIZER FROM CHECKPOINT {args.epoch_num} ###")
        optimizer.load_state_dict(torch.load(optim_fp, map_location=args.device.type))
            

    # context model
    n_features = encoder.fc.in_features  # get dimensions of fc layer
    model = SimCLR(args, encoder, n_features, args.projection_dim)
    model = model.to(args.device)
    print(model.summary())

    # optimizer / scheduler
    optimizer, scheduler = load_optimizer(args, model)

    # loss function
    if args.supervised:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = NT_Xent(args.batch_size, args.temperature, args.device)

    # DDP
    if args.nodes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu])


    writer = SummaryWriter()
    # save random init. model
    if not args.reload:
        args.current_epoch = "random"
        args.train_stage = 0
        save_model(args, model, optimizer, args.model_name)

        # write a few audio files to TensorBoard for comparison
        write_audio_tb(args, train_loader, test_loader, writer)

    # start training
    solver = Solver(model, optimizer, criterion, writer)
    validate_idx = 10
    for epoch in range(args.start_epoch, args.epochs):
        if epoch % args.checkpoint_epochs == 0:
            audio_latent_representations(args, train_loader.dataset, model, args.current_epoch, args.global_step, writer, train=True)
            audio_latent_representations(args, test_loader.dataset, model, args.current_epoch, args.global_step, writer, train=False)

        learning_rate = optimizer.param_groups[0]["lr"]

        metrics = solver.train(args, train_loader)
        for k, v in metrics.items():
            writer.add_scalar(k, v, epoch)
        writer.add_scalar("Misc/learning_rate", learning_rate, epoch)

        print(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {metrics['Loss/train']}\t lr: {round(learning_rate, 5)}"
        )

        if epoch > 0 and epoch % validate_idx == 0:
            metrics = solver.validate(args, test_loader)
            print(
                f"[Test] Epoch [{epoch}/{args.epochs}]\t Test Loss: {metrics['Loss/test']}"
            )

    ## end training
    save_model(args, model, optimizer, name=args.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLMR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    args.global_step = 0
    args.current_epoch = 0

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)
