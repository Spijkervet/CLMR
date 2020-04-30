import os
import torch
from modules import SimCLR, LogisticRegression, LARS, MLP, SampleCNN


def load_model(args, reload_model=False, name="clmr"):

    if name == "clmr":
        model = SimCLR(args)
    elif name == "supervised":
        model = SampleCNN(args)
    elif name == "eval":
        if args.mlp:
            model = MLP(args.n_features, args.n_classes)
        else:
            model = LogisticRegression(args.n_features, args.n_classes)
    else:
        raise Exception("Cannot infer model from configuration")
    
    if reload_model:
        model_path = args.model_path if name == "clmr" else args.logreg_model_path
        epoch_num = args.epoch_num if name == "clmr" else args.logreg_epoch_num
        print(f"### RELOADING {name.upper()} MODEL FROM CHECKPOINT {epoch_num} ###")
        model_fp = os.path.join(
            model_path, "{}_checkpoint_{}.tar".format(name, epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))

    model = model.to(args.device)

    scheduler = None
    if args.optimizer == "Adam" or args.lin_eval:
        print("### Using Adam optimizer ###")
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # TODO: LARS
    elif args.optimizer == "LARS":
        print("### Using LARS optimizer ###")
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    return model, optimizer, scheduler


def save_model(args, model, optimizer, name="clmr"):
    out = os.path.join(
        args.out_dir, "{}_checkpoint_{}.tar".format(name, args.current_epoch)
    )

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)
