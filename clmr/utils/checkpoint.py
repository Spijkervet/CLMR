import torch
from collections import OrderedDict


def load_encoder_checkpoint(checkpoint_path: str) -> OrderedDict:
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if "pytorch-lightning_version" in state_dict.keys():
        new_state_dict = OrderedDict(
            {
                k.replace("model.encoder.", ""): v
                for k, v in state_dict["state_dict"].items()
                if "model.encoder." in k
            }
        )
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "encoder." in k:
                new_state_dict[k.replace("encoder.", "")] = v

    new_state_dict["fc.weight"] = torch.zeros(50, 512)
    new_state_dict["fc.bias"] = torch.zeros(50)
    return new_state_dict


def load_finetuner_checkpoint(checkpoint_path: str) -> OrderedDict:
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if "pytorch-lightning_version" in state_dict.keys():
        state_dict = OrderedDict(
            {
                k.replace("model.", ""): v
                for k, v in state_dict["state_dict"].items()
                if "model." in k
            }
        )
    return state_dict
