import essentia.standard
import os
import torch
import torchvision
import argparse
import numpy as np
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

from data import get_dataset
from model import load_encoder, load_optimizer, save_model
from modules import SimCLR, SampleCNN59049, BYOL
from solver import Solver
from utils import eval_all, write_audio_tb, args_hparams, parse_args
from validation import audio_latent_representations, vision_latent_representations
from utils import parse_args

def main(gpu, args):
    rank = args.nr * args.gpus + gpu
    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(gpu)

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    # data loaders
    (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    ) = get_dataset(args, train_sampler, pretrain=True, download=args.download)
    
    encoder = load_encoder(args)
    
    # context model
    # model = BYOL(encoder, args.audio_length) # new!
    model = SimCLR(args, encoder, args.n_features, args.projection_dim)
    model = model.to(args.device)
    print(model.summary())

    # optimizer / scheduler
    optimizer, scheduler = load_optimizer(args, model)

    # DDP
    if args.nodes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu])

    writer = SummaryWriter()
    # save random init. model
    if not args.reload:
        args.current_epoch = "random"
        save_model(args, model, optimizer, args.model_name)

        # write a few audio files to TensorBoard for comparison
        write_audio_tb(args, train_loader, test_loader, writer)

    # start training
    args.current_epoch = 0
    solver = Solver(model, optimizer, writer)
    validate_idx = 10
    for epoch in range(args.start_epoch, args.epochs):
        # if epoch % validate_idx == 0:
        #     audio_latent_representations(
        #         args,
        #         train_loader.dataset,
        #         model,
        #         args.current_epoch,
        #         args.global_step,
        #         writer,
        #         train=True,
        #     )
        #     audio_latent_representations(
        #         args,
        #         test_loader.dataset,
        #         model,
        #         args.current_epoch,
        #         args.global_step,
        #         writer,
        #         train=False,
        #     )

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

        if epoch > 0 and epoch % args.checkpoint_epochs == 0:
            save_model(args, model, optimizer, name=args.model_name)

        args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer, name=args.model_name)


if __name__ == "__main__":
    args = parse_args()
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
