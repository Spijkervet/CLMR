import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import numpy as np

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from data import get_dataset
from experiment import ex
from model import load_model
from model import save_model

from utils import post_config_hook
from utils.eval import eval_all, average_precision

# metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score



def train(args, loader, simclr_model, model, criterion, optimizer, writer):
    loss_epoch = 0
    auc_epoch = 0
    accuracy_epoch = 0
    predicted_classes = torch.zeros(args.n_classes).to(args.device)
    for step, ((_, _, x), y, idx) in enumerate(loader):
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

        loss.backward()
        optimizer.step()

        predictions = output.argmax(1).detach()
        classes, counts = torch.unique(predictions, return_counts=True)
        predicted_classes[classes] += counts

        # print(output)

        # ap
        if args.domain == "audio":
            auc, acc = tagwise_auc_ap(
                y.cpu().detach().numpy(), output.cpu().detach().numpy()
            )
            auc = auc.mean()
            acc = acc.mean()
        elif args.domain == "scores":
            auc = 0
            acc = average_precision(
                y.detach().cpu().numpy(), output.detach().cpu().numpy()
            ).mean()
        else:
            raise NotImplementedError
        auc_epoch += auc
        accuracy_epoch += acc

        # acc
        # acc = (predictions == y.argmax(1)).sum().item() / y.size(0)
        # accuracy_epoch += acc

        loss_epoch += loss.item()
        if step % 100 == 0:
            print(
                f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t AUC: {auc}\t AP: {acc}"
            )

        writer.add_scalar("AUC/train_step", auc, args.global_step)
        writer.add_scalar("AP/train_step", acc, args.global_step)
        writer.add_scalar("Loss/train_step", loss, args.global_step)
        args.global_step += 1

    figure = plt.figure()
    plt.bar(range(predicted_classes.size(0)), predicted_classes.cpu().numpy())
    writer.add_figure(
        "Class_distribution/train", figure, global_step=args.current_epoch
    )
    return loss_epoch, auc_epoch, accuracy_epoch


def test(args, loader, simclr_model, model, criterion, optimizer, writer):
    loss_epoch = 0
    auc_epoch = 0
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

        predictions = output.argmax(1).detach()
        classes, counts = torch.unique(predictions, return_counts=True)
        predicted_classes[classes] += counts

        # acc = (predictions == y.argmax(1)).sum().item() / y.size(0)
        if args.domain == "audio":
            auc, acc = tagwise_auc_ap(
                y.cpu().detach().numpy(), output.cpu().detach().numpy()
            )
            auc = auc.mean()
            acc = acc.mean()
        elif args.domain == "scores":
            auc = 0
            acc = average_precision(
                y.detach().cpu().numpy(), output.detach().cpu().numpy()
            ).mean()
        else:
            raise NotImplementedError

        auc_epoch += auc
        accuracy_epoch += acc

        loss_epoch += loss.item()
        if step % 100 == 0:
            print(
                f"[Test] Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t AUC: {auc}\t AP: {acc}"
            )

    figure = plt.figure()
    plt.bar(range(predicted_classes.size(0)), predicted_classes.cpu().numpy())
    writer.add_figure("Class_distribution/test", figure, global_step=args.current_epoch)
    return loss_epoch, auc_epoch, accuracy_epoch


def solve(
    args, train_loader, test_loader, simclr_model, model, criterion, optimizer, writer
):
    for epoch in range(args.logistic_epochs):
        loss_epoch, auc_epoch, accuracy_epoch = train(
            args, train_loader, simclr_model, model, criterion, optimizer, writer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t AUC: {auc_epoch / len(train_loader)}\t AP: {accuracy_epoch / len(train_loader)}"
        )

        writer.add_scalar("AUC/train", auc_epoch / len(train_loader), epoch)
        writer.add_scalar("AP/train", accuracy_epoch / len(train_loader), epoch)
        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)

        ## testing
        loss_epoch, auc_epoch, accuracy_epoch = test(
            args, test_loader, simclr_model, model, criterion, optimizer, writer
        )
        print(
            f"[Test]\t Loss: {loss_epoch / len(test_loader)}\t AUC: {auc_epoch / len(test_loader)}\t AP: {accuracy_epoch / len(test_loader)}"
        )
        writer.add_scalar("AUC/test", auc_epoch / len(test_loader), epoch)
        writer.add_scalar("AP/test", accuracy_epoch / len(test_loader), epoch)
        writer.add_scalar("Loss/test", loss_epoch / len(test_loader), epoch)

        save_model(args, model, optimizer, name="supervised")
        args.current_epoch += 1

    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t AP: {accuracy_epoch / len(test_loader)}"
    )


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)
    args.lin_eval = True

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.batch_size = args.logistic_batch_size

    (train_loader, train_dataset, test_loader, test_dataset) = get_dataset(args)

    simclr_model, _, _ = load_model(args, reload_model=True, name="context")
    simclr_model.eval()

    args.n_features = simclr_model.n_features

    model, optimizer, _ = load_model(args, reload_model=False, name="supervised")
    model = model.to(args.device)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()  # for tags

    # initialize TensorBoard
    tb_dir = os.path.join(args.out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    args.global_step = 0
    args.current_epoch = 0

    # run training
    solve(
        args,
        train_loader,
        test_loader,
        simclr_model,
        model,
        criterion,
        optimizer,
        writer,
    )

    # eval all
    loss_epoch, auc_epoch, accuracy_epoch = eval_all(
        args, test_loader, simclr_model, model, criterion, optimizer, writer
    )
