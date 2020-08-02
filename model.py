import os
import torch
from modules import SimCLR, SampleCNN, LARS, get_resnet, Identity
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


def load_optimizer(args, model):

    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # TODO: LARS
    elif args.optimizer == "LARS":
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

    if args.reload:
        optim_fp = os.path.join(
            args.model_path,
            "{}_checkpoint_{}_optim.pt".format(args.model_name, args.epoch_num),
        )
        print(
            f"### RELOADING {args.model_name.upper()} OPTIMIZER FROM CHECKPOINT {args.epoch_num} ###"
        )
        optimizer.load_state_dict(torch.load(optim_fp, map_location=args.device.type))

    return optimizer, scheduler


def load_encoder(args, reload=False):
    # encoder
    if args.domain == "audio":
        if args.sample_rate == 22050:
            strides = [3, 3, 3, 3, 3, 3, 3, 3, 3]
        elif args.sample_rate == 16000:
            strides = [3, 3, 3, 3, 3, 3, 5, 2, 2]
        elif args.sample_rate == 8000:
            strides = [3, 3, 3, 2, 2, 4, 4, 2, 2]
        else:
            raise NotImplementedError
        
        encoder = SampleCNN(args, strides)
        print(f"### {encoder.__class__.__name__} ###")
    elif args.domain == "scores":
        encoder = get_resnet(args.resnet, pretrained=False)  # resnet
        encoder.conv1 = nn.Conv2d(
            args.image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    else:
        raise NotImplementedError
    
    args.n_features = list(encoder.children())[-1].in_features
    encoder.fc = (Identity()) # TODO rewrite this
    if reload:
        # reload model
        print(
            f"### RELOADING {args.model_name.upper()} MODEL FROM CHECKPOINT {args.epoch_num} ###"
        )
        model_fp = os.path.join(
            args.model_path,
            "{}_checkpoint_{}.pt".format(args.model_name, args.epoch_num),
        )

        # tmp workaround
        mapping = torch.load(model_fp, map_location=args.device.type)
        new_mapping = {}
        for m in mapping:
            if "conv" in m:
                new_m = m.replace("encoder.", "").split(".")
                conv_num = int(new_m[0].replace("conv", ""))
                seq_m = "sequential.{}.{}.{}".format(conv_num-1, new_m[1], new_m[2])
                new_mapping[seq_m] = mapping[m]
            elif "encoder" in m:
                new_m = m.replace("encoder.", "")
                new_mapping[new_m] = mapping[m]
        encoder.load_state_dict(new_mapping, strict=True)
    return encoder


def save_model(args, model, optimizer, name="clmr"):
    if args.model_path is not None:

        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
            
        out = os.path.join(
            args.model_path, "{}_checkpoint_{}.pt".format(name, args.current_epoch)
        )

        optim_out = os.path.join(
            args.model_path,
            "{}_checkpoint_{}_optim.pt".format(name, args.current_epoch),
        )

        # To save a DataParallel model generically, save the model.module.state_dict().
        # This way, you have the flexibility to load the model any way you want to any device you want.
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), out)
        else:
            torch.save(model.state_dict(), out)

        torch.save(optimizer.state_dict(), optim_out)
