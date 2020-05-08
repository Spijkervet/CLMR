import os
import json
import numpy as np
import torchaudio
from tqdm import tqdm
from pathlib import Path
import colorsys

def tsne_to_json(audio_length, dataset, embedding, labels):

    vis_dir = os.path.join("visualisation", "audio")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

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
    prev_track = None
    for x, y, label in tqdm(zip(x, y, labels)):
        audio_path = Path(dataset.tracks_list[int(label)][1])
        track_name = audio_path.stem # _, fp, _ = tracks_list
        c = {}
        c["coordinates"] = [float(x), float(y)]
        all_d["tsne"].append(c)
        f["names"].append(track_name)

        start_idx = idx * audio_length
        fn = os.path.join(
            f"{int(label.item())}-{track_name}-{start_idx}.mp3"
        )
        f["filenames"].append(fn)  # dirty fix


        audio_fp = os.path.join(vis_dir, fn)
        if not os.path.exists(audio_fp):
            audio, sr = torchaudio.load(audio_path)
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


def _get_colors(num_colors):
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors
