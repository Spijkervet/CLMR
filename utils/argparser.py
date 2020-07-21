import argparse
import copy
from .yaml_config_hook import yaml_config_hook

def parse_args():
    parser = argparse.ArgumentParser(description="CLMR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        typ = type(v)
        if typ == bool:
            typ = int
        parser.add_argument(f"--{k}", default=v, type=typ)


    args = parser.parse_args()
    print(args)
    exit(0)
    return args

def args_hparams(args):
    args_dict = copy.deepcopy(vars(args))
    del args_dict["device"]
    return args_dict
