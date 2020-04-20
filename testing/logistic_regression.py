import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import numpy as np

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from experiment import ex
from model import load_model
from utils import post_config_hook

# audio
from modules import LogisticRegression
from data import get_fma_loaders
from data import get_mtt_loaders

# vision
from data.vision import get_deepscores_dataloader

from utils import tagwise_auc_ap

# metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score


def average_precision(y_targets, y_preds):

    ap = []
    for by, bp in zip(y_targets, y_preds):
        ap.append(average_precision_score(by, bp))
        # acc = accuracy_score(y.argmax(1).detach().cpu().numpy(), output.argmax(1).detach().cpu().numpy())

    return np.array(ap)


def train(args, loader, simclr_model, model, criterion, optimizer, writer):
    loss_epoch = 0
    accuracy_epoch = 0
    predicted_classes = torch.zeros(args.n_classes).to(args.device)
    for step, ((_, _, x), y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        # get encoding
        with torch.no_grad():
            h, z = simclr_model(x)

        h = h.detach()
        z = z.detach()

        output = model(h)
        loss = criterion(output, y)

        predictions = output.argmax(1)
        classes, counts = torch.unique(predictions, return_counts=True)
        predicted_classes[classes] += counts

        # print(output)
        # acc = auc
        # acc, _ = tagwise_auc_ap(y.cpu().detach().numpy(), output.cpu().detach().numpy())
        # acc = acc.mean()

        # ap
        acc = average_precision(
            y.detach().cpu().numpy(), output.detach().cpu().numpy()
        ).mean()
        accuracy_epoch += acc

        # acc
        # acc = (predictions == y.argmax(1)).sum().item() / y.size(0)
        # accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        if step % 100 == 0:
            print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t AP: {acc}")

        writer.add_scalar("AP/train_step", acc, args.global_step)
        writer.add_scalar("Loss/train_step", loss, args.global_step)
        args.global_step += 1

    figure = plt.figure()
    plt.bar(range(predicted_classes.size(0)), predicted_classes.cpu().numpy())
    writer.add_figure(
        "Class_distribution/train", figure, global_step=args.current_epoch
    )
    return loss_epoch, accuracy_epoch


def test(args, loader, simclr_model, model, criterion, optimizer, writer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    predicted_classes = torch.zeros(args.n_classes).to(args.device)
    for step, ((_, _, x), y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        # get encoding
        with torch.no_grad():
            h, z = simclr_model(x)

        output = model(h)
        loss = criterion(output, y)

        predictions = output.argmax(1)
        classes, counts = torch.unique(predictions, return_counts=True)
        predicted_classes[classes] += counts

        # acc = (predictions == y.argmax(1)).sum().item() / y.size(0)
        # accuracy_epoch += acc

        acc = average_precision(
            y.detach().cpu().numpy(), output.detach().cpu().numpy()
        ).mean()
        accuracy_epoch += acc

        loss_epoch += loss.item()
        if step % 100 == 0:
            print(
                f"[Test] Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t AP: {acc}"
            )

    figure = plt.figure()
    plt.bar(range(predicted_classes.size(0)), predicted_classes.cpu().numpy())
    writer.add_figure(
        "Class_distribution/test", figure, global_step=args.current_epoch
    )

    return loss_epoch, accuracy_epoch


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)
    args.lin_eval = True

    if len(_run.observers) > 1:
        out_dir = _run.observers[1].dir
    else:
        out_dir = _run.observers[0].dir
    args.out_dir = out_dir

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.batch_size = args.logistic_batch_size

    root = "./datasets"

    # if args.dataset == "billboard":
    #     train_dataset = MIRDataset(
    #         args,
    #         os.path.join(args.data_input_dir, f"{args.dataset}_samples"),
    #         os.path.join(args.data_input_dir, f"{args.dataset}_labels/train_split.txt"),
    #         audio_length=args.audio_length,
    #         transform=AudioTransforms(args),
    #     )

    #     test_dataset = MIRDataset(
    #         args,
    #         os.path.join(args.data_input_dir, f"{args.dataset}_samples"),
    #         os.path.join(args.data_input_dir, f"{args.dataset}_labels/test_split.txt"),
    #         audio_length=args.audio_length,
    #         transform=AudioTransforms(args),
    #     )

    #     train_loader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=args.batch_size,
    #         shuffle=(train_sampler is None),
    #         drop_last=True,
    #         num_workers=args.workers,
    #         sampler=train_sampler,
    #     )

    #     test_loader = torch.utils.data.DataLoader(
    #         test_dataset,
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         drop_last=True,
    #         num_workers=args.workers,
    #     )
    # elif args.dataset == "fma":
    #     (train_loader, train_dataset, test_loader, test_dataset) = get_fma_loaders(args)
    # elif args.dataset == "mtt":
    #     (train_loader, train_dataset, test_loader, test_dataset) = get_mtt_loaders(args)
    # else:
    #     raise NotImplementedError

    (
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    ) = get_deepscores_dataloader(args)

    simclr_model, _, _ = load_model(args, train_loader, reload_model=True)
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    ## Logistic Regression
    if args.task == "tag":
        n_classes = args.num_tags
    else:
        n_classes = len(train_dataset.class_names)
        # n_classes = args.n_classes

    args.n_classes = n_classes
    print(len(train_dataset), len(test_dataset), args.n_classes)

    model = LogisticRegression(simclr_model.n_features, n_classes)
    model = model.to(args.device)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()  # for tags
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # initialize TensorBoard
    tb_dir = os.path.join(out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(
            args, train_loader, simclr_model, model, criterion, optimizer, writer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t AP: {accuracy_epoch / len(train_loader)}"
        )
        writer.add_scalar("AP/train", accuracy_epoch / len(train_loader), epoch)
        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)


        ## testing
        loss_epoch, accuracy_epoch = test(
            args, test_loader, simclr_model, model, criterion, optimizer, writer
        )
        print(
            f"[Test]\t Loss: {loss_epoch / len(test_loader)}\t AP: {accuracy_epoch / len(test_loader)}"
        )
        writer.add_scalar("AP/test", accuracy_epoch / len(test_loader), epoch)
        writer.add_scalar("Loss/test", loss_epoch / len(test_loader), epoch)
        args.current_epoch += 1

    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t AP: {accuracy_epoch / len(test_loader)}"
    )
