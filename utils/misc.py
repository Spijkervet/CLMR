import copy

def args_hparams(args):
    args_dict = copy.deepcopy(vars(args))
    del args_dict["device"]
    return args_dict
