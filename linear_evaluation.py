import essentia.standard
import torch
import argparse
import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from collections import defaultdict
import time
import urllib.request

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset
from model import load_encoder, save_model

from utils import parse_args, args_hparams, random_undersample_balanced
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


def train(args, loader, encoder, model, criterion, optimizer):
    predicted_classes = torch.zeros(args.n_classes).to(args.device)
    metrics = defaultdict(float)
    for step, (x, y) in enumerate(loader):
        # if len(x), we did not pre-compute features
        if type(x) == tuple:
            x = x[0]
            x = x.to(args.device)
            with torch.no_grad():
                x = encoder(x)
        else:
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

        if step > 0 and step % 100 == 0:
            print(
                f"[{step}/{len(loader)}]:\tLoss: {loss.item()}\tAUC_tag: {auc}\tAP_tag: {acc}"
            )

        args.global_step += 1

    # plot_predicted_classes(predicted_classes, args.current_epoch, writer, train=True)
    for k, v in metrics.items():
        metrics[k] /= len(loader)
    return metrics


def validate(args, loader, encoder, model, criterion, optimizer):
    model.eval()
    predicted_classes = torch.zeros(args.n_classes).to(args.device)
    metrics = defaultdict(float)
    for step, (x, y) in enumerate(loader):
        # if len(x), we did not pre-compute features
        if type(x) == tuple:
            x = x[0]
            x = x.to(args.device)
            with torch.no_grad():
                x = encoder(x)
        else:
            x = x.to(args.device)

        x = x.to(args.device)
        y = y.to(args.device)

        with torch.no_grad():
            output = model(x)

        loss = criterion(output, y)
        predicted_classes = get_predicted_classes(output, predicted_classes)

        if args.task == "tags" and args.dataset in ["magnatagatune", "msd"]:
            auc, acc = get_metrics(
                args.domain, y.detach().cpu().numpy(), output.detach().cpu().numpy()
            )
        else:
            predictions = output.argmax(1).detach()
            auc = 0
            acc = (predictions == y).sum().item() / y.shape[0]

        metrics["AUC_tag/test"] += auc
        metrics["AP_tag/test"] += acc
        metrics["Loss/test"] += loss.item()

        if step > 0 and step % 100 == 0:
            print(
                f"[{step}/{len(loader)}]:\tLoss: {loss.item()}\tAUC_tag: {auc}\tAP_tag: {acc}"
            )

        args.global_step += 1

    # plot_predicted_classes(predicted_classes, args.current_epoch, writer, train=True)
    for k, v in metrics.items():
        metrics[k] /= len(loader)
    
    model.train()
    return metrics



if __name__ == "__main__":
    args = parse_args()

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
    ) = get_dataset(args, pretrain=False)

    # load pre-trained encoder
    encoder = load_encoder(args, reload=True)
    encoder.eval()
    encoder = encoder.to(args.device)

    # linear eval. model
    model = torch.nn.Sequential(
        torch.nn.Linear(args.n_features, args.n_features),
        torch.nn.ReLU(),
        torch.nn.Linear(args.n_features, args.n_classes)
        )
    model = model.to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.logistic_lr, weight_decay=args.weight_decay
    )

    # set criterion, e.g. gtzan has one label per segment, MTT has multiple
    if args.dataset in ["gtzan"]:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()  # for tags

    # initialize TensorBoard
    writer = SummaryWriter()

    train_X = None
    if not os.path.exists("features.p"):
        output = input("Download (0) or compute (1) features from pre-trained network? Type \"0\" or \"1\": ")
        try:
            # download
            if int(output) == 0:
                urllib.request.urlretrieve("https://github.com/spijkervet/clmr", "features.p")
                with open("features.p", "rb") as f:
                    (train_X, train_y, test_X, test_y) = pickle.load(f)
            # compute
            elif int(output) == 1:
                print("Computing features...")
                (train_X, train_y, test_X, test_y) = get_features(
                    encoder, train_loader, test_loader, args.device
                )
            else:
                raise Exception("Invalid option")
        except Exception as e:
            print(e)
            exit(0)

        with open("features.p", "wb") as f:
            pickle.dump((train_X, train_y, test_X, test_y), f)
    else:
        with open("features.p", "rb") as f:
            (train_X, train_y, test_X, test_y) = pickle.load(f)

    if train_X is not None:
        _test_loader = test_loader # for final evaluation
        train_loader, test_loader = create_data_loaders_from_arrays(
            train_X, train_y, test_X, test_y, 2048
        )

    # start linear evaluation
    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.logistic_epochs):
        metrics = train(args, train_loader, encoder, model, criterion, optimizer)
        for k, v in metrics.items():
            writer.add_scalar(k, v, epoch)

        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {metrics['Loss/train']}\t AUC_tag: {metrics['AUC_tag/train']}\tAP_tag: {metrics['AP_tag/train']}"
        )
        
        # validate
        metrics = validate(args, test_loader, encoder, model, criterion, optimizer)
        for k, v in metrics.items():
            writer.add_scalar(k, v, epoch)

        args.current_epoch += 1

    # eval all
    metrics = eval_all(args, _test_loader, encoder, model, writer, n_tracks=None,)
    print("### Final tag/clip ROC-AUC/PR-AUC scores ###")
    for k, v in metrics.items():
        if "hparams" in k:
            print("[Test average AUC/AP]:", k, v)
        else:
            for tag, val in zip(_test_loader.dataset.tags, v):
                print(f"[Test {k}]\t\t{tag}\t{val}")