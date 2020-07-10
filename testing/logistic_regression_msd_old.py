import essentia.standard
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

from utils import post_config_hook, args_hparams, load_context_config, random_undersample_balanced
from utils.eval import (
    get_metrics,
    itemwise_auc_ap,
    tagwise_auc_ap,
    eval_all,
    average_precision,
)

# metrics
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from utils.optimizer import set_learning_rate

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


def inference(loader, context_model, model_name, n_features, device):
    feature_vector = []
    labels_vector = []

    for step, (track_id, fp, y, segment) in enumerate(loader.dataset.tracks_list_test):
        x = loader.dataset.get_full_size_audio(track_id, fp)
        x = x.to(device)
        
        y = np.tile(y, (x.shape[0], 1))

        # get encoding
        with torch.no_grad():
            if model_name == "clmr":
                h, z = context_model(x) # clmr
            else:
                z, c = context_model.model.get_latent_representations(x)  # cpc
                h = c  # use context vector
                h = h.permute(0, 2, 1)
                pooled = torch.nn.functional.adaptive_avg_pool1d(h, 1)  # one label
                pooled = pooled.permute(0, 2, 1).reshape(-1, n_features)
                h = pooled

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y)

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader.dataset.tracks_list_test)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


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


def train(args, loader, context_model, model, criterion, optimizer, writer):
    import torchaudio
    loss_epoch = 0
    auc_epoch = []
    acc_epoch = []
    predicted_classes = torch.zeros(args.n_classes).to(args.device)

    ys = []
    outputs = []
    for step, (track_id, fp, y, segment) in enumerate(loader.dataset.tracks_list_test):
        optimizer.zero_grad()
        x = loader.dataset.get_full_size_audio(fp)
        y = torch.from_numpy(np.tile(y, (x.shape[0], 1)))

        ## code check
        # torchaudio.save(f"{track_id}.mp3", x[0], sample_rate=args.sample_rate)
        # print(track_id)
        # for lidx, l in enumerate(y[0]):
        #     if l == 1:
        #         print(loader.dataset.tags[lidx])

        # if track_id > 5:
        #     exit(0)
            

        x = x.to(args.device)
        y = y.to(args.device)

        with torch.no_grad():
            h, z = context_model(x)

        h = h.detach()
        output = model(h)

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        predicted_classes = get_predicted_classes(output, predicted_classes)

        if args.task == "tags" and args.dataset in ["magnatagatune", "msd"]:
            auc = 0
            acc = 0
            ys.extend(y.cpu().detach().numpy())
            outputs.extend(output.cpu().detach().numpy())
            # auc, acc = get_metrics(args.domain, y, output)
        else:
            predictions = output.argmax(1).detach()
            auc = 0
            acc = (predictions == y).sum().item() / y.shape[0]
            

        loss_epoch += loss.item()

        if step > 0 and step % 100 == 0:
            auc, acc = get_metrics(args.domain, np.array(ys), np.array(outputs))
            print(
                f"Step [{step}/{len(loader.dataset.tracks_list_test)}]\t Loss: {loss.item()}\t AUC: {auc}\t AP: {acc}"
            )

            writer.add_scalar("AUC/train_step", auc, args.global_step)
            writer.add_scalar("AP/train_step", acc, args.global_step)
            writer.add_scalar("Loss/train_step", loss, args.global_step)
            auc_epoch.append(auc)
            acc_epoch.append(acc)

            ys = []
            outputs = []

        args.global_step += 1
    
    auc_epoch = np.array(auc_epoch).mean()
    acc_epoch = np.array(acc_epoch).mean()

    plot_predicted_classes(predicted_classes, args.current_epoch, writer, train=True)
    return loss_epoch, auc_epoch, acc_epoch


def test(args, loader, context_model, model, criterion, optimizer, writer):
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

            if args.task == "tags" and args.dataset in ["magnatagatune", "msd"]:
                auc, acc = get_metrics(args.domain, y, output)
            elif args.task == "chords":
                auc, acc = eval_chords(y, output)
            else:
                predictions = output.argmax(1).detach()
                auc = 0
                acc = (predictions == y).sum().item() / y.shape[0]

            auc_epoch += auc
            accuracy_epoch += acc
            loss_epoch += loss.item()

            # if step % 100 == 0:
            #     print(
            #         f"[Test] Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t AUC: {auc}\t AP: {acc}"
            #     )

    plot_predicted_classes(predicted_classes, args.current_epoch, writer, train=False)
    return loss_epoch, auc_epoch, accuracy_epoch


def solve(args, train_loader, val_loader, test_loader, context_model, model, criterion, optimizer, writer):
    validate_epoch = 50

    for epoch in range(args.logistic_epochs):
        loss_epoch, auc_epoch, accuracy_epoch = train(
            args, train_loader, context_model, model, criterion, optimizer, writer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t AUC: {auc_epoch}\t AP: {accuracy_epoch}"
        )

        writer.add_scalar("AUC/train", auc_epoch / len(train_loader), epoch)
        writer.add_scalar("AP/train", accuracy_epoch / len(train_loader), epoch)
        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)

        save_model(args, model, optimizer, name="supervised")

        ## testing
        if epoch > 0 and epoch % validate_epoch == 0:
            val_loss_epoch, auc_epoch, accuracy_epoch = test(
                args, val_loader, context_model, model, criterion, optimizer, writer
            )
            print(
                f"[Validation]\t Loss: {val_loss_epoch / len(test_loader)}\t AUC: {auc_epoch / len(test_loader)}\t AP: {accuracy_epoch / len(test_loader)}"
            )
            writer.add_scalar("AUC/validation", auc_epoch / len(test_loader), epoch)
            writer.add_scalar("AP/validation", accuracy_epoch / len(test_loader), epoch)
            writer.add_scalar("Loss/validation", val_loss_epoch / len(test_loader), epoch)
        
        args.current_epoch += 1

    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t AP: {accuracy_epoch / len(test_loader)}"
    )

@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)

    lrs = [0.0005] # , 0.001, 0.003, 0.004]
    epochs = [50] # , 150, 300, 500]
    for lr in lrs:
        for e in epochs:
            # load from epoch num
            print(f"LR: {lr}, Logreg_epochs: {e}")


            args = load_context_config(args)
            args.lin_eval = True
            args.at_least_one_pos = False
            
            args = post_config_hook(args, _run)

            args.logistic_lr = lr
            args.logistic_epochs = e

            args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            args.batch_size = args.logistic_batch_size

            (train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset) = get_dataset(args)

            context_model, _, _ = load_model(args, reload_model=True, name=args.model_name)
            context_model.eval()

            args.n_features = context_model.n_features

            model, _, _ = load_model(args, reload_model=False, name="supervised")
            model = model.to(args.device)

            print(model.summary())

            if not args.mlp:
                weight_decay = args.weight_decay  # sample_weight_decay()
            else:
                weight_decay = args.weight_decay  # TODO
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.logistic_lr, weight_decay=weight_decay
            )

            # set criterion, e.g. gtzan has one label per segment, MTT has multiple
            if args.dataset in ["gtzan"]:
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.BCEWithLogitsLoss()  # for tags

            # initialize TensorBoard
            writer = SummaryWriter(log_dir=args.tb_dir + f"-{lr}-{e}")

            args.global_step = 0
            args.current_epoch = 0

            print(context_model)
            print(model)
            
            # run training
            try:
                solve(
                    args,
                    train_loader,
                    val_loader,
                    test_loader,
                    context_model,
                    model,
                    criterion,
                    optimizer,
                    writer,
                )
            except KeyboardInterrupt:
                print("\n\nTerminated training, starting evaluation\n")

            # eval all
            metrics = eval_all(
                args, test_loader, context_model, model, writer, n_tracks=None,
            )
            for k, v in metrics.items():
                print("[Test]:", k, v)

            try:
                writer.add_hparams(args_hparams(args), metrics)
            except:
                pass
        
        save_model(args, model, optimizer, name="supervised")
