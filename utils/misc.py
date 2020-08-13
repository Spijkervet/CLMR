import os
import json
from datetime import datetime
import yaml
import copy

def label_to_tag(list_of_tags, label):
    with open(list_of_tags, "r") as f:
        tags = f.readlines()
    return tags[label]


def get_log_dir(root_dir, idx):
    log_dir = os.path.join(root_dir, str(idx))
    now = datetime.now()
    if os.path.exists(log_dir):
        log_dir = os.path.join(root_dir, "{}-{}".format(idx, now.strftime("%b%d_%H_%M-%S")))
    os.makedirs(log_dir)
    return log_dir


def write_args(args):
    with open(os.path.join(args.model_path, "config.yaml"), "w") as f:
        d = copy.deepcopy(vars(args))
        if "device" in d.keys():
            del d["device"]
        f.write(yaml.dump(d))
