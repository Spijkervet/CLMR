import torch
import argparse
import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from collections import defaultdict
import time
import urllib.request
import logging
import json
import copy

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# DP
from torch.nn.parallel import DataParallel

from data import get_dataset
from model import load_encoder, save_model

from utils import parse_args, args_hparams, random_undersample_balanced, get_log_dir
from utils.eval import get_metrics, get_f1_score, eval_all
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


def train(args, loader, encoder, model, criterion, optimizer, writer):
    predicted_classes = torch.zeros(args.n_classes).to(args.device)
    metrics = defaultdict(float)
    for step, (x, y) in enumerate(loader):
        # if len(x) contains the two x_i, x_j, we did not pre-compute features
        if len(x) == 2:
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

        auc = 0
        acc = 0
        f1 = 0
        if args.task == "tags" and args.dataset in ["magnatagatune", "msd"]:
            auc, acc = get_metrics(
                args.domain, y.detach().cpu().numpy(), output.detach().cpu().numpy()
            )
        elif args.dataset == "birdsong":
            f1 = get_f1_score(y.detach().cpu().numpy(), output.detach().cpu().numpy())
        else:
            predictions = output.argmax(1).detach()
            acc = (predictions == y).sum().item() / y.shape[0]

        metrics["AUC_tag/train"] += auc
        metrics["AP_tag/train"] += acc
        metrics["F1/train"] += f1
        metrics["Loss/train"] += loss.item()

        writer.add_scalar("AUC_tag/train_step", auc, args.global_step)
        writer.add_scalar("AP_tag/train_step", acc, args.global_step)
        writer.add_scalar("Loss/train_step", loss, args.global_step)

        if step > 0 and step % 100 == 0:
            logging.info(
                f"[{step}/{len(loader)}]:\tLoss: {loss.item()}\tAUC_tag: {auc}\tAP_tag: {acc}\tF1: {f1}"
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
        if len(x) == 2:
            x = x[0]
            x = x.to(args.device)
            with torch.no_grad():
                x = encoder(x)
        else:
            x = x.to(args.device)

        y = y.to(args.device)

        with torch.no_grad():
            output = model(x)

        loss = criterion(output, y)
        predicted_classes = get_predicted_classes(output, predicted_classes)
        
        auc = 0
        acc = 0
        f1 = 0
        if args.task == "tags" and args.dataset in ["magnatagatune", "msd"]:
            auc, acc = get_metrics(
                args.domain, y.detach().cpu().numpy(), output.detach().cpu().numpy()
            )
        elif args.dataset == "birdsong":
            f1 = get_f1_score(y.detach().cpu().numpy(), output.detach().cpu().numpy())
        else:
            predictions = output.argmax(1).detach()
            acc = (predictions == y).sum().item() / y.shape[0]

        metrics["AUC_tag/test"] += auc
        metrics["AP_tag/test"] += acc
        metrics["F1/test"] += f1
        metrics["Loss/test"] += loss.item()

        if step > 0 and step % 100 == 0:
            logging.info(
                f"[{step}/{len(loader)}]:\tLoss: {loss.item()}\tAUC_tag: {auc}\tAP_tag: {acc}"
            )

    # plot_predicted_classes(predicted_classes, args.current_epoch, writer, train=True)
    for k, v in metrics.items():
        metrics[k] /= len(loader)

    model.train()
    return metrics


if __name__ == "__main__":
    args = parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.batch_size = args.logistic_batch_size
    args.world_size = 1

    # Get the data loaders, without a train sampler and without pre-training (this is linear evaluation)
    (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    ) = get_dataset(args, download=args.download, pretrain=False)

    # load pre-trained encoder
    encoder = load_encoder(args, reload=True)
    encoder.eval()
    encoder = encoder.to(args.device)

    # linear eval. model
    if not args.mlp:
        model = torch.nn.Sequential(torch.nn.Linear(args.n_features, args.n_classes),)
    else:
         model = torch.nn.Sequential(
            torch.nn.Linear(args.n_features, args.n_features),
            torch.nn.ReLU(),
            torch.nn.Linear(args.n_features, args.n_classes)
        )
    model = model.to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.logistic_lr, weight_decay=args.weight_decay
    )

    if args.dataparallel:
        model = DataParallel(model)
        model = model.to(args.device)


    # set criterion, e.g. gtzan has one label per segment, MTT has multiple
    if args.dataset in ["gtzan"]:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()  # for tags

    # initialize TensorBoard
    experiment_idx = int(args.model_path.split("/")[-1])
    args.model_path = get_log_dir("results", experiment_idx)
    writer = SummaryWriter(log_dir=args.model_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_path, "output.log")),
            logging.StreamHandler(),
        ],
    )

    train_X = None
    if args.dataset == "magnatagatune" or args.perc_train_data <= 0.1:
        logging.info("Computing features...")
        (train_X, train_y, val_X, val_y, test_X, test_y) = get_features(
            encoder, train_loader, val_loader, test_loader, args.device
        )

    # The Million Song Dataset is too large to fit in memory (of most machines)
    # if args.dataset != "msd":
    #     if not os.path.exists("features.p"):
    #         output = input("Download (0) or compute (1) features from pre-trained network? Type \"0\" or \"1\": ")
    #         try:
    #             # download
    #             if int(output) == 0:
    #                 urllib.request.urlretrieve("https://github.com/spijkervet/clmr", "features.p")
    #                 with open("features.p", "rb") as f:
    #                     (train_X, train_y, val_X, val_y, test_X, test_y) = pickle.load(f)
    #             # compute
    #             elif int(output) == 1:
    #                 logging.info("Computing features...")
    #                 (train_X, train_y, val_X, val_y, test_X, test_y) = get_features(
    #                     encoder, train_loader, val_loader, test_loader, args.device
    #                 )
    #             else:
    #                 raise Exception("Invalid option")
    #         except Exception as e:
    #             logging.info(e)
    #             exit(0)

    #         with open("features.p", "wb") as f:
    #             pickle.dump((train_X, train_y, val_X, val_y, test_X, test_y), f)
    #     else:
    #         with open("features.p", "rb") as f:
    #             (train_X, train_y, val_X, val_y, test_X, test_y) = pickle.load(f)

    if train_X is not None:
        train_loader, val_loader, _ = create_data_loaders_from_arrays(
            train_X,
            train_y,
            val_X,
            val_y,
            test_X,
            test_y,
            2048,  # batch size for logistic regression (pre-computed features)
        )

    # start linear evaluation
    args.global_step = 0
    args.current_epoch = 0
    last_model = None
    last_auc = 0
    last_ap = 0
    early_stop = 0
    for epoch in range(args.logistic_epochs):
        metrics = train(
            args, train_loader, encoder, model, criterion, optimizer, writer
        )
        for k, v in metrics.items():
            writer.add_scalar(k, v, epoch)

        logging.info(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {metrics['Loss/train']}\t AUC_tag: {metrics['AUC_tag/train']}\tAP_tag: {metrics['AP_tag/train']}"
        )

        # validate
        metrics = validate(args, val_loader, encoder, model, criterion, optimizer)
        for k, v in metrics.items():
            writer.add_scalar(k, v, epoch)

        if metrics["AUC_tag/test"] < last_auc and metrics["AP_tag/test"] < last_ap:
            last_model = copy.deepcopy(model)
            early_stop += 1
            logging.info(
                "Early stop count: {}\t\t{} (best: {})\t {} (best: {})".format(
                    early_stop,
                    metrics["AUC_tag/test"],
                    last_auc,
                    metrics["AP_tag/test"],
                    last_ap,
                )
            )
        else:
            last_auc = metrics["AUC_tag/test"]
            last_ap = metrics["AP_tag/test"]
            early_stop = 0

        if early_stop >= 5:
            logging.info("Early stopping...")
            break
        args.current_epoch += 1

    if last_model is None:
        logging.info("No early stopping, using last model")
        last_model = model

    save_model(args, last_model, optimizer, name="finetuner")

    # eval all
    metrics = eval_all(args, test_loader, encoder, last_model, writer, n_tracks=None)
    logging.info("### Final tag/clip ROC-AUC/PR-AUC scores ###")
    m = {}
    for k, v in metrics.items():
        if "hparams" in k:
            logging.info(f"[Test average AUC/AP]: {k}, {v}")
            m[k] = v
        else:
            for tag, val in zip(test_loader.dataset.tags, v):
                logging.info(f"[Test {k}]\t\t{tag}\t{val}")
                m[k + "/" + tag] = val

    with open(os.path.join(args.model_path, "results.json"), "w") as f:
        json.dump(m, f)
