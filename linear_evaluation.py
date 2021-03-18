import os
import argparse
from torch.utils.data import DataLoader
from torchaudio_augmentations import Compose, RandomResizedCrop
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


from clmr.datasets import get_dataset
from clmr.data import ContrastiveDataset
from clmr.evaluation import evaluate
from clmr.models import SampleCNN
from clmr.modules import ContrastiveLearning, LinearEvaluation, PlotSpectogramCallback
from clmr.utils import yaml_config_hook, load_encoder_checkpoint


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.accelerator = None

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("That checkpoint does not exist")

    train_transform = [RandomResizedCrop(n_samples=args.audio_length)]
    
    # ------------
    # dataloaders
    # ------------
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")

    contrastive_train_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=(1, args.audio_length),
        transform=Compose(train_transform),
    )

    contrastive_test_dataset = ContrastiveDataset(
        test_dataset,
        input_shape=(1, args.audio_length),
        transform=Compose(train_transform),
    )

    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True,
    )

    test_loader = DataLoader(
        contrastive_test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=False,
    )

    # ------------
    # encoder
    # ------------
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=args.supervised,
        out_dim=train_dataset.n_classes,
    )

    n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer
    
    state_dict = load_encoder_checkpoint(args.checkpoint_path)
    encoder.load_state_dict(state_dict)

    cl = ContrastiveLearning(args, encoder)
    cl.eval()
    cl.freeze()

    l = LinearEvaluation(
        args,
        cl.encoder,
        hidden_dim=n_features,
        output_dim=train_dataset.n_classes,
    )

    if args.linear_checkpoint_path:
        l = l.load_from_checkpoint(
            args.linear_checkpoint_path,
            encoder=cl.encoder,
            hidden_dim=n_features,
            output_dim=train_dataset.n_classes,
        )
    else:
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[PlotSpectogramCallback()],
            logger=TensorBoardLogger(
                "runs", name="CLMRv2-eval-{}".format(args.dataset)
            )
        )
        trainer.fit(l, train_loader, test_loader)

    results = evaluate(l.encoder, l.model, contrastive_test_dataset, args.audio_length)
    print(results)