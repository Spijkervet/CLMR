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

from modules.pytorchtools import EarlyStopping
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
    for step, ((x, _), y, _) in enumerate(loader):
        # for xb in x:
        #     import torchaudio
        #     torchaudio.save("audio.wav", xb, sample_rate=22050)
        #     exit(0)

        x = x.to(device)

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
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(context_model, train_loader, val_loader, test_loader, model_name, n_features, device):
    train_X, train_y = inference(train_loader, context_model, model_name, n_features, device)
    val_X, val_y = inference(val_loader, context_model, model_name, n_features, device)
    test_X, test_y = inference(test_loader, context_model, model_name, n_features, device)
    return train_X, train_y, val_X, val_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    # X_train, X_test = normalize_dataset(X_train, X_test)

    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).type(torch.float)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )
    
    val = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val).type(torch.float)
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test).type(torch.float)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader


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

        if args.task == "tags" and args.dataset in ["magnatagatune"]:
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
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t AUC: {auc}\t AP: {acc}"
        #     )

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

            if args.task == "tags" and args.dataset in ["magnatagatune"]:
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


def solve(args, train_loader, val_loader, test_loader, model, criterion, optimizer, writer):
    validate_epoch = 1
    max_train_stages = 1

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
            val_loss_epoch, auc_epoch, accuracy_epoch = test(
                args, val_loader, model, criterion, optimizer, writer
            )
            print(
                f"[Validation]\t Loss: {val_loss_epoch / len(test_loader)}\t AUC: {auc_epoch / len(test_loader)}\t AP: {accuracy_epoch / len(test_loader)}"
            )
            writer.add_scalar("AUC/validation", auc_epoch / len(test_loader), epoch)
            writer.add_scalar("AP/validation", accuracy_epoch / len(test_loader), epoch)
            writer.add_scalar("Loss/validation", val_loss_epoch / len(test_loader), epoch)
        

    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t AP: {accuracy_epoch / len(test_loader)}"
    )
    args.train_stage += 1

@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)

    for i in range(args.epoch_num, args.epoch_num+1, 1):

        # load from epoch num

        args = load_context_config(args)
        args.lin_eval = True
        args.at_least_one_pos = False
        
        args = post_config_hook(args, _run)
        args.epoch_num = i

        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.batch_size = args.logistic_batch_size

        (train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset) = get_dataset(args)

        context_model, _, _ = load_model(args, reload_model=True, name=args.model_name)
        context_model.eval()

        args.n_features = context_model.n_features

        model, _, _ = load_model(args, reload_model=False, name="eval")
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
        writer = SummaryWriter(log_dir=args.tb_dir + "-" + str(i))

        args.global_step = 0
        args.current_epoch = 0

        print(context_model)
        print(model)

        # create features from pre-trained model
        if not os.path.exists("features.p"):
            print("### Creating features from pre-trained context model ###")
            (train_X, train_y, val_X, val_y, test_X, test_y) = get_features(
                context_model, train_loader, val_loader, test_loader, args.model_name, args.n_features, args.device
            )
            pickle.dump(
                (train_X, train_y, val_X, val_y, test_X, test_y), open("features.p", "wb"), protocol=4
            )
        else:
            print("### Loading features ###")
            (train_X, train_y, val_X, val_y, test_X, test_y) = pickle.load(open("features.p", "rb"))

        if args.perc_train_data < 1.0:
            print("Train dataset size:", len(train_X))
            train_X, train_y = random_undersample_balanced(train_X, train_y, args.perc_train_data)
            print("Undersampled train dataset size:", len(train_X))


        from sklearn.neural_network import MLPClassifier
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import SVC
        # model = SVC(kernel='linear', verbose=True, max_iter=10)
        model = MLPClassifier(hidden_layer_sizes=(512), verbose=10, 
            solver='sgd', learning_rate='constant', learning_rate_init=0.003)

        print('Fitting model..')
        model.fit(train_X, train_y)
        print('Evaluating model..')

        print('Predict labels on evaluation data')


        pred_array = model.predict(test_X)
        score = model.score(test_X, test_y)

        # agreggating same ID: majority voting
        y_true = []
        y_pred = []
        id_array = np.array([track_id for track_id, _, _, _ in test_loader.dataset.tracks_list])
        id_array = id_array[:len(pred_array)] # drop last
        for track_id, _, label, _ in test_loader.dataset.tracks_list[:len(pred_array)]:
            avg = np.mean(pred_array[np.where(id_array == track_id)], axis=0)
            y_pred.append(avg)
            y_true.append(label.numpy())

        from sklearn.metrics import roc_auc_score, log_loss

        print('raw score', score)
        # conf_matrix = confusion_matrix(y_true, y_pred)
        # print(conf_matrix)
        # acc = accuracy_score(y_true, y_pred)
        # print(acc)

        roc_auc = roc_auc_score(y_true, y_pred, average="macro")
        pr_auc = average_precision_score(y_true, y_pred, average="macro")
        print(roc_auc, pr_auc)
        exit(0)