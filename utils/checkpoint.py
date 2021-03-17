import torch
from collections import OrderedDict


def load_encoder_checkpoint(checkpoint_path: str) -> OrderedDict:
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = {}
    for k, v in state_dict.items():
        if "encoder." in k:
            new_state_dict[k.replace("encoder.", "")] = v
    new_state_dict["fc.weight"] = torch.zeros(50, 512)
    new_state_dict["fc.bias"] = torch.zeros(50)
    return new_state_dict