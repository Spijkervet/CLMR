import os
import torch
import torchvision
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter

apex = False
try:
    from apex import amp

    apex = True
except ImportError:
    print(
        "Install the apex package from https://www.github.com/nvidia/apex to use fp16 for training"
    )

from data import get_dataset
from datasets.utils.prepare_dataset import prepare_dataset
from torchvision.utils import save_image


from model import load_model, save_model
from modules import NT_Xent
from utils import mask_correlated_samples, post_config_hook, tensor_to_audio
from validation import audio_latent_representations, vision_latent_representations


#### pass configuration
from experiment import ex

TMP_DIR = ".tmp"


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j, _), _) in enumerate(train_loader):
        if not os.path.exists(TMP_DIR):
            os.makedirs(TMP_DIR)

        # see transformations
        # for idx, (i, j) in enumerate(zip(x_i, x_j)):
        #     save_image(i, f"{TMP_DIR}/x_i{idx}_{args.current_epoch}.png")
        #     save_image(j, f"{TMP_DIR}/x_j{idx}_{args.current_epoch}.png")

        # hear transformations
        for idx, (i, j) in enumerate(zip(x_i, x_j)):
            tensor_to_audio(f"{TMP_DIR}/x_i{idx}_{args.current_epoch}.mp3", i, sr=args.sample_rate)
            tensor_to_audio(f"{TMP_DIR}/x_j{idx}_{args.current_epoch}.mp3", j, sr=args.sample_rate)

        optimizer.zero_grad()
        x_i = x_i.to(args.device)
        x_j = x_j.to(args.device)

        # positive pair, with encoding
        h_i, z_i = model(x_i)
        h_j, z_j = model(x_j)

        loss = criterion(z_i, z_j)

        if apex and args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        if step % 1 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
        loss_epoch += loss.item()
        args.global_step += 1
    return loss_epoch


def test(args, loader, model, criterion, writer):
    model.eval()
    loss_epoch = 0
    with torch.no_grad():
        for step, ((x_i, x_j, _), _) in enumerate(loader):
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)

            # positive pair, with encoding
            h_i, z_i = model(x_i)
            h_j, z_j = model(x_j)

            loss = criterion(z_i, z_j)

            if step % 1 == 0:
                print(f"Step [{step}/{len(loader)}]\t Test Loss: {loss.item()}")

            loss_epoch += loss.item()
    model.train()
    return loss_epoch


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.lin_eval = False  # first, pre-train, after that, lin. evaluation

    root = "./datasets"

    train_sampler = None

    # prepare_dataset(args)
    (train_loader, train_dataset, test_loader, test_dataset) = get_dataset(args)

    model, optimizer, scheduler = load_model(args, reload_model=args.reload)
    print(model)

    tb_dir = os.path.join(args.out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    mask = mask_correlated_samples(args)
    criterion = NT_Xent(args.batch_size, args.temperature, mask, args.device)

    # save random init. model
    args.current_epoch = "random"
    save_model(args, model, optimizer)

    args.global_step = 0
    args.current_epoch = 0
    validate_idx = 50
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]

        if epoch % validate_idx == 0:
            if args.domain == "audio":
                audio_latent_representations(
                    args,
                    train_loader.dataset,
                    model,
                    args.current_epoch,
                    0,
                    args.global_step,
                    writer,
                    train=True,
                )
                audio_latent_representations(
                    args,
                    test_loader.dataset,
                    model,
                    args.current_epoch,
                    0,
                    args.global_step,
                    writer,
                    train=False,
                )
            elif args.domain == "scores":
                vision_latent_representations(
                    args,
                    train_loader.dataset,
                    model,
                    optimizer,
                    args.current_epoch,
                    0,
                    args.global_step,
                    writer,
                    train=True,
                )
                vision_latent_representations(
                    args,
                    test_loader.dataset,
                    model,
                    optimizer,
                    args.current_epoch,
                    0,
                    args.global_step,
                    writer,
                    train=False,
                )
            else:
                raise NotImplementedError

        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
        writer.add_scalar("Misc/learning_rate", lr, epoch)
        print(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
        )

        # validate
        if epoch % validate_idx == 0:
            print("Validation")
            test_loss_epoch = test(args, test_loader, model, criterion, writer)
            writer.add_scalar("Loss/test", test_loss_epoch / len(test_loader), epoch)

        if scheduler:
            scheduler.step()

        if epoch % 10 == 0:
            save_model(args, model, optimizer)

        args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)
