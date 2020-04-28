import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


def itemwise_auc_ap(y, pred):
    """ Annotation : item-wise(row wise) calculation """
    n_songs = y.shape[0]
    auc = []
    ap = []
    for i in range(n_songs):
        if y[i].sum() != 0:
            auc.append(roc_auc_score(y[i], pred[i], average="macro"))
            ap.append(average_precision_score(y[i], pred[i], average="macro"))
    return np.array(auc), np.array(ap)


def tagwise_auc_ap(y, pred):
    """ tag-wise (col wise) calculation
    input:
        y: batches of true labels (batch_size x num_tags)
        pred: batches of logits (batch_size x  num_tags)
    output:
        auc: auc score for
        """
    n_tags = y.shape[1]
    auc = []
    ap = []
    for i in range(n_tags):
        if y[:, i].sum() != 0:
            auc.append(roc_auc_score(y[:, i], pred[:, i], average="macro"))
            ap.append(average_precision_score(y[:, i], pred[:, i]))
    return np.array(auc), np.array(ap)


def average_precision(y_targets, y_preds):
    ap = []
    for by, bp in zip(y_targets, y_preds):
        ap.append(average_precision_score(by, bp))
        # acc = accuracy_score(y.argmax(1).detach().cpu().numpy(), output.argmax(1).detach().cpu().numpy())

    return np.array(ap)


def eval_all(
    args, loader, simclr_model, model, writer, n_tracks=None
):
    model.eval()
    predicted_classes = torch.zeros(args.n_classes).to(args.device)
    pred_array = []
    id_array = []

    # sub-sample if n_tracks is not none, else eval whole dataset
    if n_tracks:
        ids = np.random.choice(len(loader.dataset.tracks_list), n_tracks)
        tracks = [
            (track_id, fp, label)
            for track_id, fp, label in loader.dataset.tracks_list
            if track_id in ids
        ]
    else:
        tracks = loader.dataset.tracks_list[:n_tracks]
        n_tracks = len(tracks)

    with torch.no_grad():
        # run all audio through model and make prediction array
        for step, (track_id, fp, y) in enumerate(tracks):
            x = loader.dataset.get_full_size_audio(track_id, fp)

            x = x.to(args.device)
            y = y.to(args.device)

            # get encoding
            with torch.no_grad():
                h, z = simclr_model(x)

            output = model(h)
            predictions = output.argmax(1).detach()
            classes, counts = torch.unique(predictions, return_counts=True)
            predicted_classes[classes] += counts

            # create array of predictions and ids
            for b in output:
                pred_array.append(b.detach().cpu().numpy())
                id_array.append(track_id)

            if step % 100 == 0:
                print(f"[Test] Step [{step}/{n_tracks}]")

    # normalise pred_array acc. ids
    y_pred = []
    y_true = []
    pred_array = np.array(pred_array)
    id_array = np.array(id_array)
    for track_id, _, label in tracks:
        avg = np.mean(pred_array[np.where(id_array == track_id)], axis=0)
        y_pred.append(avg)
        y_true.append(label.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    auc, ap = tagwise_auc_ap(y_true, y_pred)

    auc = auc.mean()
    ap = ap.mean()

    figure = plt.figure()
    plt.bar(range(predicted_classes.size(0)), predicted_classes.cpu().numpy())
    writer.add_figure(
        "Class_distribution/test_all", figure, global_step=args.current_epoch
    )
    model.train()
    return auc, ap
