import torch
import argparse
import os
import numpy as np

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset
from experiment import ex
from model import load_model

from utils import post_config_hook
from utils.eval import eval_all
from utils.youtube import download_yt
from datasets.utils.resample import convert_samplerate

import matplotlib.pyplot as plt

@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args.lin_eval = True

    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.batch_size = args.logistic_batch_size

    (train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset) = get_dataset(args)

    context_model, _, _ = load_model(args, reload_model=True, name=args.model_name)
    context_model.eval()

    args.n_features = context_model.n_features

    model, _, _ = load_model(args, reload_model=True, name="supervised")
    model = model.to(args.device)

    print(context_model)
    print(model)

    # initialize TensorBoard
    writer = SummaryWriter(log_dir=args.tb_dir)

    args.input_file = "yt.mp3"
    args.current_epoch = 0
    
    download_yt(args.audio_url, args.input_file, args.sample_rate)

    conv_fn = f"{args.input_file}_{args.sample_rate}"
    convert_samplerate(args.input_file, conv_fn, args.sample_rate)

    yt_audio = train_dataset.get_audio(conv_fn)
    
    # to mono
    yt_audio = yt_audio.mean(axis=0).reshape(1, -1)

    # split into equally sized tensors of args.audio_length
    chunks = torch.split(yt_audio, args.audio_length, dim=1)
    chunks = chunks[:-1] # remove last one, since it's not a complete segment of audio_length
    predicted_classes = torch.zeros(args.n_classes).to(args.device)

    with torch.no_grad():
        taggram = []
        for idx, x in enumerate(chunks):
            x = x.to(args.device)

            # normalise
            x = train_dataset.normalise_audio(x)

            # add batch dim
            x = x.unsqueeze(0)         
            h, z = context_model(x)

            output = model(h)
            output = torch.nn.functional.softmax(output, dim=1)
            output = output.squeeze(0) # remove batch dim
            taggram.append(output.cpu().detach().numpy())
    
    taggram = np.array(taggram)


    ## global figure settings
    plt.rcParams["figure.figsize"] = (20,10) # set size of the figures
    fontsize = 12 # set figures font size
    in_length = yt_audio.shape[1] / args.sample_rate
    
    ## figure
    fig, ax = plt.subplots()
    ax.title.set_text('Taggram')
    ax.title.set_fontsize(fontsize)
    ax.set_xlabel('(seconds)', fontsize=fontsize)

    ## y-axis
    tags = train_dataset.tags
    y_pos = np.arange(len(tags))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tags, fontsize=fontsize-1)

    ## x-axis
    x_pos = np.arange(0, taggram.shape[0], 10)
    ax.set_xticks(x_pos)
    x_label = [int(x * (args.audio_length / args.sample_rate)) for x in x_pos]
    ax.set_xticklabels(x_label, fontsize=6)


    plt.imshow(taggram.T, interpolation=None, aspect="auto")
    plt.savefig("inference.png")


    # mean of taggram
    fig, ax = plt.subplots()
    tags_likelihood_mean = np.mean(taggram, axis=0) # averaging the Taggram through time
    # title
    ax.title.set_text('Tags likelihood (mean of the taggram)')
    ax.title.set_fontsize(fontsize)

    # y-axis title
    ax.set_ylabel('(likelihood)', fontsize=fontsize)

    # y-axis
    ax.set_ylim((0, 1))
    ax.tick_params(axis="y", labelsize=fontsize)

    # x-axis
    ax.tick_params(axis="x", labelsize=fontsize-1)
    pos = np.arange(len(tags))
    ax.set_xticks(pos)
    ax.set_xticklabels(tags, rotation=90)

    # depict song-level tags likelihood
    ax.bar(pos, tags_likelihood_mean)
    plt.savefig("inference_likelihood.png")

    N = 3
    tags_likelihoods = [(t, l) for t, l in zip(tags, tags_likelihood_mean)]
    top_N = sorted(tags_likelihoods, key=lambda x: x[1], reverse=True)[:N]
    print(top_N)

    os.remove(args.input_file)
    os.remove(conv_fn)

    
    # # eval all
    # metrics = eval_all(
    #     args,
    #     test_loader,
    #     context_model,
    #     model,
    #     writer,
    #     n_tracks=None,
    # )
     
    # for k, v in metrics.items():
    #     print(f"[Test]: {k}: {v}")
