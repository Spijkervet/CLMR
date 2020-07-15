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
    encoder.eval()
    if model:
        model.eval()

    predicted_classes = torch.zeros(args.n_classes).to(args.device)
    pred_array = []
    y_true = []
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
        tracks = loader.dataset.index
        tracks = [[track_id, clip_id, segment, fp, label] for track_id, clip_id, segment, fp, label in tracks if segment == 0] # get_full_size_audio
        n_tracks = len(tracks)

    with torch.no_grad():
        # run all audio through model and make prediction array
        for step, (track_id, clip_id, segment, fp, label) in enumerate(tracks):
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
            
            # average prediction of all segments
            pred_array.append(output.mean(axis=0).detach().cpu().numpy())
            y_true.append(label.numpy())

            if step % 1000 == 0:
                print(f"[Test] Step [{step}/{n_tracks}]")
                
    # normalise pred_array acc. ids
    y_pred = np.array(pred_array)
    y_true = np.array(y_true)

    metrics = {}
    if args.dataset in ["magnatagatune", "msd"]:
        auc, ap = tagwise_auc_ap(y_true, y_pred)
        clip_auc, clip_ap = itemwise_auc_ap(y_true, y_pred)
        # for TensorBoard
        metrics["all/test_tag_auc"] = auc
        metrics["all/test_tag_ap"] = ap
        metrics["all/test_clip_auc"] = clip_auc
        metrics["all/test_clip_ap"] = clip_ap

        metrics["hparams/test_tag_auc_mean"] = auc.mean()
        metrics["hparams/test_tag_ap_mean"] = ap.mean()
        metrics["hparams/test_clip_auc_mean"] = clip_auc.mean()
        metrics["hparams/test_clip_ap_mean"] = clip_ap.mean()
    else:
        acc = accuracy_score(y_true, y_pred.argmax(1))
        metrics["hparams/test_accuracy"] = acc

    figure = plt.figure()
    plt.bar(range(predicted_classes.size(0)), predicted_classes.cpu().numpy())
    writer.add_figure(
        "Class_distribution/test_all", figure, global_step=args.current_epoch
    )

    encoder.train()
    if model:
        model.train()
    return metrics
