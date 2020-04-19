import torch
import torchvision
import torchvision.transforms as transforms
import argparse

from experiment import ex
from model import load_model
from utils import post_config_hook

from modules import LogisticRegression
from data import get_fma_loaders
from data import get_mtt_loaders

from utils import tagwise_auc_ap


def train(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, ((x, _), y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        # get encoding
        with torch.no_grad():
            h, z = simclr_model(x)

        output = model(h)
        loss = criterion(output, y)

        # predicted = output.argmax(1)
        # print(output)
        # acc = auc
        acc, _ = tagwise_auc_ap(y.cpu().detach().numpy(), output.cpu().detach().numpy())
        acc = acc.mean()

        # acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        if step % 100 == 0:
            print(
                f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
            )

    return loss_epoch, accuracy_epoch


def test(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, ((x_i, x_j), y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        # get encoding
        with torch.no_grad():
            h, z = simclr_model(x)

        output = model(h)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)
    args.lin_eval = True

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.batch_size = args.logistic_batch_size

    root = "./datasets"

    if args.dataset == "billboard":
        train_dataset = MIRDataset(
            args,
            os.path.join(args.data_input_dir, f"{args.dataset}_samples"),
            os.path.join(args.data_input_dir, f"{args.dataset}_labels/train_split.txt"),
            audio_length=args.audio_length,
            transform=AudioTransforms(args),
        )

        test_dataset = MIRDataset(
            args,
            os.path.join(args.data_input_dir, f"{args.dataset}_samples"),
            os.path.join(args.data_input_dir, f"{args.dataset}_labels/test_split.txt"),
            audio_length=args.audio_length,
            transform=AudioTransforms(args),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            drop_last=True,
            num_workers=args.workers,
            sampler=train_sampler,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.workers,
        )
    elif args.dataset == "fma":
        (train_loader, train_dataset, test_loader, test_dataset) = get_fma_loaders(args)
    elif args.dataset == "mtt":
        (train_loader, train_dataset, test_loader, test_dataset) = get_mtt_loaders(args)
    else:
        raise NotImplementedError

    simclr_model, _, _ = load_model(args, train_loader, reload_model=True)
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    ## Logistic Regression
    if args.task == "tag":
        n_classes = args.num_tags
    else:
        n_classes = args.n_classes

    model = LogisticRegression(simclr_model.n_features, n_classes)
    model = model.to(args.device)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()  # for tags
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(
            args, train_loader, simclr_model, model, criterion, optimizer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}"
        )

    # final testing
    loss_epoch, accuracy_epoch = test(
        args, test_loader, simclr_model, model, criterion, optimizer
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}"
    )
