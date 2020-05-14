import os
import torch
from modules import SimCLR, LogisticRegression, LARS, MLP, SampleCNN
from modules.cpc import CPCModel

def cpc_model(args):
    strides = [5, 3, 2, 2, 2, 2, 2]
    filter_sizes = [10, 6, 4, 4, 4, 2, 2]
    padding = [2, 2, 2, 2, 1, 1, 1]
    genc_hidden = 512
    gar_hidden = 256

    model = CPCModel(
        args,
        strides=strides,
        filter_sizes=filter_sizes,
        padding=padding,
        genc_hidden=genc_hidden,
        gar_hidden=gar_hidden,
    )
    return model




def load_model(args, reload_model=False, name="clmr"):
    if name == "clmr":
        model = SimCLR(args)
    elif name == "cpc":
        model = cpc_model(args)
    elif name == "eval":
        if args.mlp:
            model = MLP(args.n_features, args.n_classes)
        else:
            model = LogisticRegression(args.n_features, args.n_classes)
    else:
        raise Exception("Cannot infer model from configuration")
    
    if reload_model:
        model_path = args.model_path if name != "eval" else args.logreg_model_path
        epoch_num = args.epoch_num if name != "eval" else args.logreg_epoch_num
        print(f"### RELOADING {name.upper()} MODEL FROM CHECKPOINT {epoch_num} ###")

        model_fp = os.path.join(
            model_path, "{}_checkpoint_{}.tar".format(name, epoch_num)
        )

<<<<<<< HEAD
        if not os.path.exists(model_fp):
            model_fp = os.path.join(
                model_path, "{}_checkpoint_{}.tar".format(name, epoch_num)
            )

        
        strict = True
        if args.transfer or name == "cpc":
=======
        strict = True
        if args.transfer:
>>>>>>> e998801a2c38fc3e304326734657aa4a03a31a9a
            strict = False
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type), strict=strict)

    model = model.to(args.device)

    scheduler = None
    if args.optimizer == "Adam" or args.lin_eval:
        print("### Using Adam optimizer ###")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate
        )
    elif args.optimizer == "SGD":
        print("### Using SGD optimizer ###")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True
        )
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

    if args.supervised:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.global_lr_decay, patience=2, verbose=True)

    if reload_model and not args.transfer:
        optim_fp = os.path.join(
            model_path, "{}_checkpoint_{}_optim.tar".format(name, epoch_num)
        )
        if os.path.exists(optim_fp):
            print(f"### RELOADING {name.upper()} OPTIMIZER FROM CHECKPOINT {epoch_num} ###")
            optimizer.load_state_dict(torch.load(optim_fp, map_location=args.device.type))
            
    model.train()
    return model, optimizer, scheduler


def save_model(args, model, optimizer, name="clmr"):
    if args.out_dir is not None:
        out = os.path.join(
            args.out_dir, "{}_checkpoint_{}.tar".format(name, args.current_epoch)
        )

        optim_out = os.path.join(
            args.out_dir, "{}_checkpoint_{}_optim.tar".format(name, args.current_epoch)
        )

        # To save a DataParallel model generically, save the model.module.state_dict().
        # This way, you have the flexibility to load the model any way you want to any device you want.
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), out)
        else:
            torch.save(model.state_dict(), out)

        torch.save(optimizer.state_dict(), optim_out)
