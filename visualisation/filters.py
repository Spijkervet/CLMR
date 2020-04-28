import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from data import get_dataset
from model import load_model
from utils import post_config_hook


#### pass configuration
from experiment import ex

class SaveFeatures():
    def __init__(self, args, module):
        self.args = args
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True).to(self.args.device)
    def close(self):
        self.hook.remove()


class FilterVisualizer():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.upscaling_steps = 0

    def visualize(self, x, layer, filter, lr=3e-4, opt_steps=20, blur=None):
        print(x)
        activations = SaveFeatures(self.args, list(self.model.children())[layer])  # register hook
        optimizer = torch.optim.Adam([x], lr=lr)
        for n in range(opt_steps):
            optimizer.zero_grad()
            self.model(x)
            loss = -activations.features[0, filter].mean()
            loss.backward()
            optimizer.step()
        
        self.output = x
        print(self.output)
        #     self.output = img
        #     sz = int(self.upscaling_factor * sz)  # calculate new image size
        #     img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)  # scale image up
        #     if blur is not None: img = cv2.blur(img,(blur,blur))  # blur image to reduce high frequency patterns
        # self.save(layer, filter)
        activations.close()


from viztools import viz_act_val, viz_cnn_filter

@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.lin_eval = True

    (train_loader, train_dataset, test_loader, test_dataset) = get_dataset(args)

    model, optimizer, scheduler = load_model(args, reload_model=True)
    model = model.eval() # set in evaluation mode (dropout, bn, etc.)
    print(model)

    args.global_step = 0
    args.current_epoch = 0
    validate_idx = 50

    input_audio = np.random.uniform(-1, 1, (1, 1, args.audio_length))
    input_audio = torch.from_numpy(input_audio).float().to(args.device)
    print(input_audio.shape)
    print(input_audio)

    fv = FilterVisualizer(args, model)
    fv.visualize(input_audio, layer=0, filter=0)