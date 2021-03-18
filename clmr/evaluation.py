import torch
from tqdm import tqdm
from sklearn import metrics

from clmr.data import ContrastiveDataset

def evaluate(encoder, finetuned_head, test_dataset, audio_length: int, gpu: bool=False) -> dict:
    est_array = []
    gt_array = []

    if gpu:
        encoder = encoder.to("cuda:0")
        if finetuned_head:
            finetuned_head = finetuned_head.to("cuda:0")
    
    encoder.eval()
    finetuned_head.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(contrastive_test_dataset))):
            _, label = contrastive_test_dataset[idx]
            batch = contrastive_test_dataset.concat_clip(idx, audio_length)
            batch = batch.to("cuda:0")

            output = encoder(batch)
            if finetuned_head:
                output = finetuned_head(output)
                
            output = torch.nn.functional.softmax(output, dim=1)
            track_prediction = output.mean(dim=0)
            est_array.append(track_prediction)
            gt_array.append(label)


    if args.dataset in ["magnatagatune"]:
        est_array = torch.stack(est_array, dim=0).cpu().numpy()
        gt_array = torch.stack(gt_array, dim=0).cpu().numpy()
        roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average="macro")
        return {
            "PR-AUC": pr_aucs
            "ROC-AUC": roc_aucs,
        }

    accuracy = metrics.accuracy_score(gt_array, est_array) 
    return {
        "Accuracy": accuracy
    }
    