import numpy as np
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


def eval_all(args, loader, simclr_model, model, criterion, optimizer, writer):
    model.eval()
    predicted_classes = torch.zeros(args.n_classes).to(args.device)
    pred_array = []
    id_array = []
    N_TRACKS = len(loader.dataset.tracks_list)
    for step, (track_id, fp, y) in enumerate(loader.dataset.tracks_list[:N_TRACKS]):
        x = loader.dataset.get_full_size_audio(track_id, fp)

        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        # get encoding
        with torch.no_grad():
            h, z = simclr_model(x)

        output = model(h)

        # create array of predictions and ids
        for b in output:
            pred_array.append(b.detach().cpu().numpy())
            id_array.append(track_id)
        
        print(f"[Test] Step [{step}/{N_TRACKS}]")

    # normalise pred_array acc. ids
    y_pred = []
    y_true = []
    pred_array = np.array(pred_array)
    id_array = np.array(id_array)
    for track_id, _, label in loader.dataset.tracks_list[:N_TRACKS]:
        avg = np.mean(pred_array[np.where(id_array == track_id)], axis=0)
        y_pred.append(avg)
        y_true.append(label.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    auc, ap = tagwise_auc_ap(y_true, y_pred)

    auc = auc.mean()
    ap = ap.mean()

    print(f"[Test]: ROC-AUC {auc}")
    print(f"[Test]: PR-AUC {ap}")

    figure = plt.figure()
    plt.bar(range(predicted_classes.size(0)), predicted_classes.cpu().numpy())
    writer.add_figure("Class_distribution/test", figure, global_step=args.current_epoch)
    return auc, ap
