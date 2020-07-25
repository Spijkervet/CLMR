import sys
import os
import torch
import torchvision
import numpy as np
import logging


from torch.utils.tensorboard import SummaryWriter

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel, DataParallel

# custom modules
from data import get_dataset
from model import load_encoder, load_optimizer, save_model
from modules import SimCLR, BYOL, NT_Xent
from modules.sync_batchnorm import convert_model
from solver import Solver
from utils import LogFile, eval_all, write_audio_tb, parse_args, get_log_dir, write_args
from validation import audio_latent_representations


def main(gpu, args):
    args.rank = args.nr * args.gpus + gpu
    
    # dataparallel splits the data across the batch dimension
    if args.dataparallel:
        args.batch_size *= args.num_gpus

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # data loaders
    (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    ) = get_dataset(args, pretrain=True, download=args.download)

    encoder = load_encoder(args)

    # context model
    # model = BYOL(encoder, args.audio_length) # new!
    model = SimCLR(args, encoder, args.n_features, args.projection_dim)
    model.apply(model.initialize)
    model = model.to(args.device)
    logging.info(model.summary())

    if args.reload:
        model_fp = os.path.join(
            args.model_path,
            "{}_checkpoint_{}.pt".format(args.model_name, args.epoch_num),
        )
        logging.info(
            f"### RELOADING {args.model_name.upper()} MODEL FROM CHECKPOINT {args.epoch_num} ###"
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))

    # optimizer / scheduler
    optimizer, scheduler = load_optimizer(args, model)

    if not args.supervised:
        criterion = NT_Xent(args.batch_size, args.temperature, args.device, args.world_size)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    # DDP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(model, device_ids=[gpu])
    model = model.to(args.device)

    writer = SummaryWriter(log_dir=args.model_path)
    # save random init. model
    if not args.reload:
        args.current_epoch = "random"
        save_model(args, model, optimizer, args.model_name)

        # write a few audio files to TensorBoard for comparison
        # write_audio_tb(args, train_loader, test_loader, writer)

    # start training
    args.current_epoch = args.start_epoch
    solver = Solver(model, optimizer, criterion, writer)
    validate_idx = 1
    for epoch in range(args.start_epoch, args.epochs):
        if epoch % validate_idx == 0:
            audio_latent_representations(
                args,
                train_loader.dataset,
                model,
                args.current_epoch,
                args.global_step,
                writer,
                train=True,
            )
            audio_latent_representations(
                args,
                test_loader.dataset,
                model,
                args.current_epoch,
                args.global_step,
                writer,
                train=False,
            )

        learning_rate = optimizer.param_groups[0]["lr"]
        metrics = solver.train(args, train_loader)
        for k, v in metrics.items():
            writer.add_scalar(k, v, epoch)
        writer.add_scalar("Misc/learning_rate", learning_rate, epoch)

        logging.info(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {metrics['Loss/train']}\t lr: {round(learning_rate, 5)}"
        )

        if epoch > 0 and epoch % validate_idx == 0:
            metrics = solver.validate(args, test_loader)
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)

            logging.info(
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
    os.environ["MASTER_PORT"] = "5000"

    args.model_path = get_log_dir("./logs", args.id)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    args.global_step = 0
    args.current_epoch = 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_path, 'output.log')),
            logging.StreamHandler()
        ]
    )

    write_args(args)

    if args.nodes > 1:
        logging.info(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)
