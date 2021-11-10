"""
This script will extract a pre-trained CLMR PyTorch model to an ONNX model.
"""

import argparse
import os
import torch
from collections import OrderedDict
from copy import deepcopy
from clmr.models import SampleCNN, Identity
from clmr.utils import load_encoder_checkpoint, load_finetuner_checkpoint


def convert_encoder_to_onnx(
    encoder: torch.nn.Module, test_input: torch.Tensor, fp: str
) -> None:
    input_names = ["audio"]
    output_names = ["representation"]

    torch.onnx.export(
        encoder,
        test_input,
        fp,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--finetuner_checkpoint_path", type=str, required=True)
    parser.add_argument("--n_classes", type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("That encoder checkpoint does not exist")

    if not os.path.exists(args.finetuner_checkpoint_path):
        raise FileNotFoundError("That linear model checkpoint does not exist")

    # ------------
    # encoder
    # ------------
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=False,
        out_dim=args.n_classes,
    )

    n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer

    state_dict = load_encoder_checkpoint(args.checkpoint_path, args.n_classes)
    encoder.load_state_dict(state_dict)
    encoder.eval()

    # ------------
    # linear model
    # ------------
    state_dict = load_finetuner_checkpoint(args.finetuner_checkpoint_path)
    encoder.fc.load_state_dict(
        OrderedDict({k.replace("0.", ""): v for k, v in state_dict.items()})
    )

    encoder_export = deepcopy(encoder)
    # set last fully connected layer to an identity function:
    encoder_export.fc = Identity()

    batch_size = 1
    channels = 1
    audio_length = 59049
    test_input = torch.randn(batch_size, 1, audio_length)

    convert_encoder_to_onnx(encoder, test_input, "clmr_sample-cnn.onnx")
    convert_encoder_to_onnx(
        encoder_export, test_input, "clmr_encoder_only_sample-cnn.onnx"
    )
