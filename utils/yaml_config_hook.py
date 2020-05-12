import os
import yaml
import json
import argparse

def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg

def post_config_hook(args, _run):
    if len(_run.observers) > 1:
        out_dir = _run.observers[1].dir
    elif len(_run.observers) == 1:
        out_dir = _run.observers[0].dir
    else:
        out_dir = "logs/test" 

    args.out_dir = out_dir

    # create TensorBoard name
    lin_txt = ""
    if args.lin_eval:
        lin_txt = "-lineval"

    if args.out_dir:
        tb_str = f"{args.domain}-{args.model_name}-{args.sample_rate}-{args.batch_size}bs-{lin_txt}"
        tb_dir = os.path.join(args.out_dir, tb_str)
        args.tb_dir = tb_dir
        if not os.path.exists(args.tb_dir):
            os.makedirs(args.tb_dir)
    return args

def load_context_config(args):
    dataset = args.dataset
    context_model_path = args.model_path
    epoch_num = args.epoch_num
    logistic_epochs = args.logistic_epochs
    mlp = args.mlp
    perc_train_data = args.perc_train_data
    logistic_lr = args.logistic_lr

    json_config = os.path.join(context_model_path, "config.json")
    context_args = json.load(open(json_config, "r"))
    new_args = argparse.Namespace(**context_args)

    new_args.dataset = dataset
    new_args.model_path = context_model_path
    new_args.epoch_num = epoch_num
    new_args.logistic_epochs = logistic_epochs
    new_args.mlp = mlp
    new_args.perc_train_data = perc_train_data
    new_args.logistic_lr = logistic_lr
    return new_args