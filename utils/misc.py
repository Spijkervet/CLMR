import copy

def args_hparams(args):
    args_dict = copy.deepcopy(vars(args))
    del args_dict["device"]
    return args_dict

def label_to_tag(list_of_tags, label):
    with open(list_of_tags, "r") as f:
        tags = f.readlines()
    return tags[label]