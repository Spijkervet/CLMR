import torch
from tqdm import tqdm
from sklearn import metrics

from clmr.data import ContrastiveDataset

def evaluate(encoder, finetuned_head, test_dataset, audio_length: int) -> dict:

    contrastive_test_dataset = ContrastiveDataset(
        test_dataset,
        input_shape=(1, audio_length),
        transform=None,
    )

    est_array = []
    gt_array = []
    encoder = encoder.to("cuda:0")
    finetuned_head = finetuned_head.to("cuda:0")
    
    encoder.eval()
    finetuned_head.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(contrastive_test_dataset))):
            _, label = contrastive_test_dataset[idx]
            batch = contrastive_test_dataset.concat_clip(idx, audio_length)
            batch = batch.to("cuda:0")

            h0 = encoder(batch)
            output = finetuned_head(h0)
            output = torch.nn.functional.softmax(output, dim=1)
            track_prediction = output.mean(dim=0)
            est_array.append(track_prediction)
            gt_array.append(label)

            # for l, tag in zip(label.reshape(-1), test_dataset.tags):
            #     if l:
            #         print("Ground truth:", tag)

            # for p, tag in zip(track_prediction, test_dataset.tags):
            #     if p:
            #         print("Predicted:", p, tag)
            # torchaudio.save("{}.wav".format(idx), batch.cpu().reshape(-1), sample_rate=22050)
            # exit()


    est_array = torch.stack(est_array, dim=0).cpu().numpy()
    gt_array = torch.stack(gt_array, dim=0).cpu().numpy()

    roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
    pr_aucs = metrics.average_precision_score(gt_array, est_array, average="macro")

    return {
        "ROC-AUC": roc_aucs,
        "PR-AUC": pr_aucs
    }