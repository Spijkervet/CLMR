import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.chords import chords, chromatic_scale

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


def eval_all(args, loader, simclr_model, model, writer, n_tracks=None):
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


def is_fifth(tonic_p, mode_p, tonic_l, mode_l):
    """
    Calculate if the tonic of the prediction is the fifth of the target (matching modes)
    """
    return (tonic_p == fifths[tonic_l]) and (mode_p == mode_l)


def is_relative(tonic_p, mode_p, tonic_l, mode_l):
    """
    Calculate if modes differ and either:
    a) the predicted mode is minor and the predicted tonic is 3 semitones below the target, or 
    b) the predicted mode is major and the predicted tonic is 3 semitones above the target
    """

    if mode_p != mode_l:
        return False

    chrom_p = chromatic_scale[tonic_p]
    chrom_l = chromatic_scale[tonic_l]
    if mode_p == "min":
        if chrom_p == chrom_l - 3:
            return True
    if mode_p == "maj":
        if chrom_p == chrom_l + 3:
            return True

    return False


def is_parallel(tonic_p, mode_p, tonic_l, mode_l):
    """
    Calculate if modes differ but the predicted tonic matches the target.
    """
    return (mode_p != mode_l) and (tonic_p == tonic_l)


def evaluate_key_mirex(predictions, labels):
    """
    MIREX weighted key evaluation
    """

    metrics = {}
    metrics["correct"] = (predictions == labels).sum().item()  # correct classifications

    # if the tonic of the prediction is the fifth of the target (or vice versa), and modes correspond.
    metrics["fifth"] = 0  #
    metrics["relative"] = 0
    metrics["parallel"] = 0
    metrics["other"] = 0

    key_names = list(chords.keys())
    for p, l in zip(predictions, labels):
        key_p, key_l = p.item(), l.item()

        key_p_name = key_names[key_p]
        key_l_name = key_names[key_l]
        tonic_p, mode_p = key_p_name.split(":")
        tonic_l, mode_l = key_l_name.split(":")

        if is_fifth(tonic_p, mode_p, tonic_l, mode_l):
            metrics["fifth"] += 1
            # print('fifth', key_p, tonic_p, mode_p, key_l, tonic_l, mode_l)
        elif is_relative(tonic_p, mode_p, tonic_l, mode_l):
            metrics["relative"] += 1
            # print('rel', key_p, tonic_p, mode_p, key_l, tonic_l, mode_l)
        elif is_parallel(tonic_p, mode_p, tonic_l, mode_l):
            metrics["parallel"] += 1
            # print('parallel', key_p, tonic_p, mode_p, key_l, tonic_l, mode_l)
        else:
            metrics["other"] += 1

    weighted = (
        metrics["correct"]
        + 0.5 * metrics["fifth"]
        + 0.3 * metrics["relative"]
        + 0.2 * metrics["parallel"]
    )

    metrics["other"] = labels.size(0) - metrics["other"]
    weighted = weighted / labels.size(0)
    metrics = {k: v / labels.size(0) for k, v in metrics.items()}
    return weighted, metrics
