import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from data import get_dataset
from model import load_encoder
from scripts.datasets.resample import resample
from utils import parse_args, download_yt

def save_taggram(yt_audio, title, sr, audio_length, taggram, tags, fp):
    taggram = np.array(taggram)

    ## global figure settings
    plt.rcParams["figure.figsize"] = (20, 10)  # set size of the figures
    fontsize = 12  # set figures font size
    in_length = yt_audio.shape[1] / sr

    ## figure
    fig, ax = plt.subplots()
    ax.title.set_text(title)
    ax.title.set_fontsize(fontsize)
    ax.set_xlabel("(seconds)", fontsize=fontsize)

    ## y-axis
    tags = tags
    y_pos = np.arange(len(tags))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tags, fontsize=fontsize - 1)

    ## x-axis
    x_pos = np.arange(0, taggram.shape[0], 10)
    ax.set_xticks(x_pos)
    x_label = [int(x * (audio_length / sr)) for x in x_pos]
    ax.set_xticklabels(x_label, fontsize=fontsize)

    plt.imshow(taggram.T, interpolation=None, aspect="auto")
    plt.tight_layout()
    plt.savefig(fp)
    print("Saving taggram in taggram.png...")
    

if __name__ == "__main__":
    args = parse_args("./config/config.yaml")

    if args.audio_url is None:
        raise Exception("args.audio_url is not provided (link to YouTube video)")

    args.world_size = 1
    args.supervised = 0
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data loaders
    (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    ) = get_dataset(args, pretrain=False, download=args.download)

    # load pre-trained encoder
    encoder = load_encoder(args, reload=True)
    encoder.eval()
    encoder = encoder.to(args.device)

    model = None
    if not args.supervised:
        finetuned_head = torch.nn.Sequential(
            torch.nn.Linear(args.n_features, args.n_classes)
        )

        finetuned_head.load_state_dict(
            torch.load(
                os.path.join(
                    args.finetune_model_path,
                    f"finetuner_checkpoint_{args.finetune_epoch_num}.pt",
                ),
             map_location=args.device
            )
        )
        finetuned_head = finetuned_head.to(args.device)

    tmp_input_file = "yt.mp3"
    args.current_epoch = 0

    print("Downloading and converting YouTube video...")
    video_title, video_id = download_yt(args.audio_url, tmp_input_file, args.sample_rate)

    conv_fn = f"{tmp_input_file}_{args.sample_rate}"
    resample(tmp_input_file, conv_fn, args.sample_rate)

    yt_audio = process_wav(args.sample_rate, conv_fn, False)
    yt_audio = torch.from_numpy(yt_audio)
    yt_audio = yt_audio.reshape(1, -1) # to mono

    # split into equally sized tensors of args.audio_length
    chunks = torch.split(yt_audio, args.audio_length, dim=1)
    chunks = chunks[
        :-1
    ]  # remove last one, since it's not a complete segment of audio_length

    print("Starting inference...")
    with torch.no_grad():
        taggram = []
        for idx, x in enumerate(chunks):
            x = x.to(args.device)

            # normalise
            if train_dataset.mean:
                x = train_dataset.normalise_audio(x)

            # add batch dim
            x = x.unsqueeze(0)
            h = encoder(x)

            output = finetuned_head(h)
            output = torch.nn.functional.softmax(output, dim=1)
            output = output.squeeze(0)  # remove batch dim
            taggram.append(output.cpu().detach().numpy())

    print("Cleaning up mp3...")
    os.remove(tmp_input_file)
    os.remove(conv_fn)

    save_taggram(yt_audio, video_title, args.sample_rate, args.audio_length, taggram, train_dataset.tags, "taggram.png")