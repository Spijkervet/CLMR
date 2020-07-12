from .get_dataloader import get_audio_dataloader

from data.vision import get_deepscores_dataloader 
from data.vision import get_universal_dataloader

# from scripts.datasets.prepare_dataset import prepare_dataset

def get_dataset(args, pretrain=True, download=False):
    val_loader = None
    val_dataset = None
    if args.domain == "audio":
        (train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset) = get_audio_dataloader(args, pretrain, download)
    elif args.domain == "scores":
        if args.dataset == "deepscores":
            (train_loader, train_dataset, test_loader, test_dataset) = get_deepscores_dataloader(args)
        elif args.dataset == "universal":
            (train_loader, train_dataset, test_loader, test_dataset) = get_universal_dataloader(args)
        else:
            raise NotImplementedError
    return train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset