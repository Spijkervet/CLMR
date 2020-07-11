import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def get_metrics(domain, y, output):
    if domain == "audio":
        auc, acc = tagwise_auc_ap(
            y, output
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


def eval_all(args, loader, encoder, model, writer, n_tracks=None):
    if model:
        model.eval()

    predicted_classes = torch.zeros(args.n_classes).to(args.device)
    pred_array = []
    id_array = []

    # sub-sample if n_tracks is not none, else eval whole dataset
    if n_tracks:
        ids = np.random.choice(len(loader.dataset.racks_list), n_tracks)
        tracks = [
            (track_id, fp, label)
            for track_id, fp, label in loader.dataset.racks_list
            if track_id in ids
        ]
    else:
        tracks = loader.dataset.tracks_list
        n_tracks = len(tracks)

    with torch.no_grad():
        # run all audio through model and make prediction array
        for step, (track_id, fp, y, _) in enumerate(tracks):
            x = loader.dataset.get_full_size_audio(fp)
            x = x.to(args.device)

            # get encoding
            if encoder:
                with torch.no_grad():
                    h = encoder(x) # clmr

            if not args.supervised:
                output = model(h)
            else:
                output = h
                
            predictions = output.argmax(1).detach()
            classes, counts = torch.unique(predictions, return_counts=True)
            predicted_classes[classes] += counts.float()
            
            # create array of predictions and ids
            for b in output:
                pred_array.append(b.detach().cpu().numpy())
                id_array.append(track_id)

            if step % 1000 == 0:
                print(f"[Test] Step [{step}/{n_tracks}]")
                
    # normalise pred_array acc. ids
    y_pred = []
    y_true = []
    pred_array = np.array(pred_array)
    id_array = np.array(id_array)
    for track_id, _, label, _ in tracks:
        # average over track
        avg = np.mean(pred_array[np.where(id_array == track_id)], axis=0)
        y_pred.append(avg)

        if isinstance(label, torch.Tensor):
            y_true.append(label.numpy())
        else:
            y_true.append(label)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    metrics = {}
    if args.dataset in ["magnatagatune", "msd"]:
        auc, ap = tagwise_auc_ap(y_true, y_pred)
        clip_auc, clip_ap = itemwise_auc_ap(y_true, y_pred)
        metrics["hparams/test_tag_auc"] = auc.mean()
        metrics["hparams/test_tag_ap"] = ap.mean()
        metrics["hparams/test_clip_auc"] = clip_auc.mean()
        metrics["hparams/test_clip_ap"] = clip_ap.mean()
        # metrics["all/auc"] = auc
        # metrics["all/ap"] = ap
    else:
        acc = accuracy_score(y_true, y_pred.argmax(1))
        metrics["hparams/test_accuracy"] = acc

    figure = plt.figure()
    plt.bar(range(predicted_classes.size(0)), predicted_classes.cpu().numpy())
    writer.add_figure(
        "Class_distribution/test_all", figure, global_step=args.current_epoch
    )

    if model:
        model.train()
    return metrics
