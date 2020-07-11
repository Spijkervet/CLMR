import essentia.standard
import torch
import argparse
import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from collections import defaultdict

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset
from model import load_encoder, save_model

from utils import yaml_config_hook, args_hparams, random_undersample_balanced
from utils.eval import get_metrics, eval_all
from features import get_features, create_data_loaders_from_arrays

def get_predicted_classes(output, predicted_classes):
    predictions = output.argmax(1).detach()
    classes, counts = torch.unique(predictions, return_counts=True)
    predicted_classes[classes] += counts.float()
    return predicted_classes


def plot_predicted_classes(predicted_classes, epoch, writer, train):
    train_test = "train" if train else "test"
    figure = plt.figure()
    plt.bar(range(predicted_classes.size(0)), predicted_classes.cpu().numpy())
    writer.add_figure(f"Class_distribution/{train_test}", figure, global_step=epoch)


def train(args, loader, model, criterion, optimizer):
    predicted_classes = torch.zeros(args.n_classes).to(args.device)
    metrics = defaultdict(float)
    for step, (x, y) in enumerate(loader):

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_classes = get_predicted_classes(output, predicted_classes)

        if args.task == "tags" and args.dataset in ["magnatagatune", "msd"]:
            auc, acc = get_metrics(
                args.domain, y.detach().cpu().numpy(), output.detach().cpu().numpy()
            )
        else:
            predictions = output.argmax(1).detach()
            auc = 0
            acc = (predictions == y).sum().item() / y.shape[0]

        metrics["AUC_tag/train"] += auc
        metrics["AP_tag/train"] += acc
        metrics["Loss/train"] += loss.item()
        args.global_step += 1

    # plot_predicted_classes(predicted_classes, args.current_epoch, writer, train=True)
    for k, v in metrics.items():
        metrics[k] /= len(loader)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLMR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.batch_size = args.logistic_batch_size

    # Get the data loaders, without a train sampler and without pre-training (this is linear evaluation)
    (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    ) = get_dataset(args, train_sampler=None, pretrain=False)

    encoder = load_encoder(args, reload=True)
    encoder.eval()
    encoder = encoder.to(args.device)

    model = torch.nn.Sequential(torch.nn.Linear(args.n_features, args.n_classes))
    model = model.to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.logistic_lr  # , weight_decay=weight_decay
    )

    # set criterion, e.g. gtzan has one label per segment, MTT has multiple
    if args.dataset in ["gtzan"]:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()  # for tags

    # initialize TensorBoard
    writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0

    (train_X, train_y, test_X, test_y) = get_features(
        encoder, train_loader, test_loader, args.device
    )

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.logistic_batch_size
    )

    for epoch in range(args.logistic_epochs):
        metrics = train(args, arr_train_loader, model, criterion, optimizer)
        for k, v in metrics.items():
            writer.add_scalar(k, v, epoch)

        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {metrics['Loss/train']}\t AUC_tag: {metrics['AUC_tag/train']}\tAP_tag: {metrics['AP_tag/train']}"
        )

        args.current_epoch += 1

    # eval all
    metrics = eval_all(args, test_loader, encoder, model, writer, n_tracks=None,)
    for k, v in metrics.items():
        print("[Test]:", k, v)

