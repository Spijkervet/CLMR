# a bit messy :)
import os
import argparse
import json
import torch
import numpy as np
import colorsys
import torchaudio
from tqdm import tqdm

from collections import defaultdict

from validation.audio.latent_representations import tsne

from data.mirdataset import track_index, default_indexer


def _get_colors(num_colors):
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


args = argparse.Namespace()
args.dataset = "billboard"
fp = os.path.join("./datasets/audio/", args.dataset, "labels", "unlabeled_split.txt")
sample_rate = 44100

tracks_index = track_index(args)
tracks_list, tracks_dict = default_indexer(fp, tracks_index, sr=sample_rate)


features = torch.load("./logs/audio/billboard/9/tsne/features-8200-0.pt")
labels = torch.load("./logs/audio/billboard/9/tsne/labels-8200-0.pt")

print("Loaded features, processing TSNE")

embedding = tsne(features)
x, y = embedding[:, 0], embedding[:, 1]


# between [0, 1]
x = (x - np.min(x)) / np.ptp(x)
y = (y - np.min(y)) / np.ptp(y)


all_d = {}
all_d["tsne"] = []
f = {}
f["names"] = []
f["filenames"] = []
f["colors"] = []

idx = 0
audio_length = 118099
prev_track = None
for x, y, label in tqdm(zip(x, y, labels)):
    track_name = tracks_index[str(int(label.item()))].title
    c = {}
    c["coordinates"] = [float(x), float(y)]
    all_d["tsne"].append(c)
    f["names"].append(track_name)

    track = tracks_index[str(int(label.item()))]
    year = track.chart_date.split("-")[0]

    start_idx = idx * audio_length
    fn = os.path.join(
        f"{int(label.item())}-{track_name}-{track.artist}-{year}-{start_idx}.mp3"
    )
    f["filenames"].append(fn)  # dirty fix

    audio_fp = os.path.join("visualisation", "audio", fn)
    if not os.path.exists(audio_fp):
        audio, sr = torchaudio.load(track.audio_path)
        audio = audio[:, start_idx : start_idx + audio_length]

        torchaudio.save(audio_fp, audio, sample_rate=sr)

    idx += 1

    if label != prev_track:
        idx = 0
        prev_track = label.item()


all_d["names"] = f["names"]
all_d["filenames"] = f["filenames"]

color_d = {}
unique_names = list(set(all_d["names"]))
colors = _get_colors(len(unique_names))
for k, v in zip(unique_names, colors):
    color_d[k] = v


# all unique tracks
for k in all_d["names"]:
    f["colors"].append(color_d[k])

all_d["colors"] = f["colors"]

with open("./visualisation/data.json", "w") as f:
    json.dump(all_d, f, ensure_ascii=False, indent=4)
