import torch
import argparse
import os
import numpy as np
import pickle

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from data import get_dataset
from experiment import ex
from model import load_model
from model import save_model

from utils import post_config_hook
from utils.eval import (
    itemwise_auc_ap,
    tagwise_auc_ap,
    eval_all,
    average_precision,
    evaluate_key_mirex,
)

# metrics
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def normalize_dataset(X_train, X_test):
    scaler = StandardScaler()
    print("Standard Scaling Normalizer")
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def sample_weight_decay():
    # We selected the l2 regularization parameter from a range of 45 logarithmically spaced values between 10−6 and 105
    weight_decay = np.logspace(-6, 5, num=45, base=10.0)
    weight_decay = np.random.choice(weight_decay)
    print("Sampled weight decay:", weight_decay)
    return weight_decay


def inference(loader, context_model, device):
    feature_vector = []
    labels_vector = []
    for step, ((_, _, x), y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, z = context_model(x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(context_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, context_model, device)
    test_X, test_y = inference(test_loader, context_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    # X_train, X_test = normalize_dataset(X_train, X_test)

    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).type(torch.float)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test).type(torch.float)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def get_predicted_classes(output, predicted_classes):
    predictions = output.argmax(1).detach()
    classes, counts = torch.unique(predictions, return_counts=True)
    predicted_classes[classes] += counts
    return predicted_classes


def get_metrics(domain, y, output):
    if domain == "audio":
        auc, acc = tagwise_auc_ap(
            y.cpu().detach().numpy(), output.cpu().detach().numpy()
        )
        auc = auc.mean()
        acc = acc.mean()
    elif domain == "scores":
        auc = 0
        acc = average_precision(
            y.detach().cpu().numpy(), output.detach().cpu().numpy()
        ).mean()
    else:
        raise NotImplementedError
    return auc, acc


def plot_predicted_classes(predicted_classes, epoch, writer, train):
    train_test = "train" if train else "test"
    figure = plt.figure()
    plt.bar(range(predicted_classes.size(0)), predicted_classes.cpu().numpy())
    writer.add_figure(f"Class_distribution/{train_test}", figure, global_step=epoch)


def eval_chords(y, output):
    auc, acc = itemwise_auc_ap(y.detach().cpu().numpy(), output.detach().cpu().numpy())
    return auc.mean(), acc.mean()


def train(args, loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    auc_epoch = 0
    accuracy_epoch = 0
    predicted_classes = torch.zeros(args.n_classes).to(args.device)
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)

        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        predicted_classes = get_predicted_classes(output, predicted_classes)

        if args.task == "tags":
            auc, acc = get_metrics(args.domain, y, output)
        elif args.task == "chords":
            auc, acc = eval_chords(y, output)
            # predictions = []
            # labels = []
            # for o, l in zip(output, y):
            #     num_chords = (l == 1).sum()
            #     _, preds = torch.topk(o, num_chords)
            #     _, chord_nums = torch.topk(l, num_chords)
            #     predictions.extend(preds)
            #     labels.extend(chord_nums)
            # predictions = torch.stack(predictions)
            # labels = torch.stack(labels)
            # weighted, metrics = evaluate_key_mirex(predictions, labels)
            # auc = 0
            # acc = weighted

        auc_epoch += auc
        accuracy_epoch += acc
        loss_epoch += loss.item()

        if step % 100 == 0:
            print(
                f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t AUC: {auc}\t AP: {acc}"
            )

        writer.add_scalar("AUC/train_step", auc, args.global_step)
        writer.add_scalar("AP/train_step", acc, args.global_step)
        writer.add_scalar("Loss/train_step", loss, args.global_step)
        args.global_step += 1

    plot_predicted_classes(predicted_classes, args.current_epoch, writer, train=True)
    return loss_epoch, auc_epoch, accuracy_epoch


def test(args, loader, model, criterion, optimizer, writer):
    model.eval()
    loss_epoch = 0
    auc_epoch = 0
    accuracy_epoch = 0
    predicted_classes = torch.zeros(args.n_classes).to(args.device)
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            x = x.to(args.device)
            y = y.to(args.device)

            output = model(x)
            loss = criterion(output, y)

            predicted_classes = get_predicted_classes(output, predicted_classes)

            if args.task == "tags":
                auc, acc = get_metrics(args.domain, y, output)
            elif args.task == "chords":
                auc, acc = eval_chords(y, output)

            auc_epoch += auc
            accuracy_epoch += acc
            loss_epoch += loss.item()

            if step % 100 == 0:
                print(
                    f"[Test] Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t AUC: {auc}\t AP: {acc}"
                )

    plot_predicted_classes(predicted_classes, args.current_epoch, writer, train=False)
    return loss_epoch, auc_epoch, accuracy_epoch


def solve(args, train_loader, test_loader, model, criterion, optimizer, writer):

    validate_epoch = 1
    for epoch in range(args.logistic_epochs):
        loss_epoch, auc_epoch, accuracy_epoch = train(
            args, train_loader, model, criterion, optimizer, writer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t AUC: {auc_epoch / len(train_loader)}\t AP: {accuracy_epoch / len(train_loader)}"
        )

        writer.add_scalar("AUC/train", auc_epoch / len(train_loader), epoch)
        writer.add_scalar("AP/train", accuracy_epoch / len(train_loader), epoch)
        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)

        save_model(args, model, optimizer, name="supervised")
        args.current_epoch += 1

        ## testing
        if epoch % validate_epoch == 0:
            loss_epoch, auc_epoch, accuracy_epoch = test(
                args, test_loader, model, criterion, optimizer, writer
            )
            print(
                f"[Test]\t Loss: {loss_epoch / len(test_loader)}\t AUC: {auc_epoch / len(test_loader)}\t AP: {accuracy_epoch / len(test_loader)}"
            )
            writer.add_scalar("AUC/test", auc_epoch / len(test_loader), epoch)
            writer.add_scalar("AP/test", accuracy_epoch / len(test_loader), epoch)
            writer.add_scalar("Loss/test", loss_epoch / len(test_loader), epoch)

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

    context_model, _, _ = load_model(args, reload_model=True, name="context")
    context_model.eval()

    args.n_features = context_model.n_features

    model, _, _ = load_model(args, reload_model=False, name="supervised")
    model = model.to(args.device)

    if not args.mlp:
        weight_decay = args.weight_decay # sample_weight_decay()
    else:
        weight_decay = args.weight_decay # TODO
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.logistic_lr, weight_decay=weight_decay
    )

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()  # for tags

    # initialize TensorBoard
    tb_dir = os.path.join(args.out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    args.global_step = 0
    args.current_epoch = 0

    print(context_model)
    print(model)

    # create features from pre-trained model
    if not os.path.exists("features.p"):
        print("### Creating features from pre-trained context model ###")
        (train_X, train_y, test_X, test_y) = get_features(
            context_model, train_loader, test_loader, args.device
        )
        pickle.dump((train_X, train_y, test_X, test_y), open("features.p", "wb"))
    else:
        print("### Loading features ###")
        (train_X, train_y, test_X, test_y) = pickle.load(open("features.p", "rb"))
    
    print("Train dataset size:", len(train_X))
    train_indices = np.random.choice(len(train_X), int(len(train_X) * args.perc_train_data), replace=False)
    train_X = train_X[train_indices]
    train_y = train_y[train_indices]
    print("Train dataset size:", len(train_X))

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X,
        train_y,
        test_X,
        test_y,
        len(test_loader.dataset) # args.logistic_batch_size
    )


    # run training
    solve(
        args, arr_train_loader, arr_test_loader, model, criterion, optimizer, writer,
    )

    # eval all
    auc, ap = eval_all(args, test_loader, context_model, model, writer, n_tracks=None,)

    print(f"[Test]: ROC-AUC {auc}")
    print(f"[Test]: PR-AUC {ap}")
