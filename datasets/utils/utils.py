import sys
import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pickle

from datasets.utils.resample import convert_samplerate
from utils.chords import (
    chords,
    keys,
    get_chord_at_interval,
    chord_to_label,
    key_to_label,
    estimate_mode,
    mode_to_label,
)


def resample_tracks(args, tracks):
    new_tracks = []
    largest_class = None
    new_labels = defaultdict(list)
    for t in tracks:
        new_labels[t.label].append(t)
        if largest_class is not None:
            if len(new_labels[t.label]) > len(new_labels[largest_class]):
                largest_class = t.label
        else:
            largest_class = t.label

    num_largest_class = len(new_labels[largest_class])
    for k, v in new_labels.items():
        new_labels[k] = resample(
            new_labels[k], replace=True, n_samples=num_largest_class, random_state=args.seed,
        )
        new_tracks.extend(new_labels[k])
    return new_tracks


def add_tracks(
    args, tracks, audio_length_sec, data_dir, label_fp, train=True,
):
    audio_len_seconds = 0
    means = []
    stds = []
    chroma_means = []
    chroma_stds = []
    class_weights = np.zeros(args.n_classes, dtype=np.int16)
    for t in tqdm(tracks):
        audio_path = t.audio_path
        fn, ext = os.path.splitext(audio_path)
        sr_audio_path = os.path.join(
            os.path.dirname(audio_path), f"{fn}_{str(args.sample_rate)}{ext}"
        )


        if t.mode is None:  # discard song if mode is unknown
            continue
        
        # chroma = t.chroma
        if os.path.exists(audio_path):
            if not os.path.exists(sr_audio_path):
                print(f"Converting {t.title}")
                convert_samplerate(audio_path, sr_audio_path, args.sample_rate)

            audio, sr = torchaudio.load(sr_audio_path, normalization=False)
            audio = audio.mean(axis=0).reshape(1, -1)  # to mono
            assert (
                sr == args.sample_rate
            ), "Sample rate is not consistent throughout the dataset"

            # # discard last part that is not a full 5 seconds
            ms = sr / audio_length_sec
            max_length = int(audio.size(1) // ms * ms)
            # samples = np.arange(ms, max_length, ms)

            samples_splits = np.arange(0, max_length, sr * audio_length_sec)
            samples = torch.split(audio, sr * audio_length_sec, dim=1)

            # transform track_chords
            for sample_idx, sample in enumerate(samples):
                if sample.size(1) < args.audio_length:
                    continue
                
                start_idx = samples_splits[sample_idx]
                end_idx = samples_splits[sample_idx] + sample.size(1)

                track_dir = os.path.join(data_dir, t.track_id)
                if not os.path.exists(track_dir):
                    os.makedirs(track_dir)

                """
                label for task
                """
                torchaudio.save(
                    os.path.join(
                        track_dir,
                        "{}-{}-{}-{}.wav".format(sample_idx, start_idx, end_idx, t.label),
                    ),
                    sample,
                    sample_rate=sr,
                )

     
                mean = sample.mean()
                std = sample.std()


                audio_len_seconds += sample.size(1) / sr
                means.append(mean)
                stds.append(std)


                class_weights[t.label] += 1

                # chroma_tensor = torch.zeros(sample.size(1))
                # for idx in range(len(chroma) - 1):
                #     onset = chroma[idx][1]
                #     chroma1 = chroma[idx][2:14]
                #     onset_samples = int(chroma[idx][1] * args.sample_rate)
                #     next_onset_samples = int(chroma[idx + 1][1] * args.sample_rate)
                #     # TODO
                #     chroma_tensor[onset_samples:next_onset_samples] = float(chroma1.argmax())

                #     if next_onset_samples > chroma_tensor.size(0):
                #         break

                # torch.save(
                #     chroma_tensor,
                #     os.path.join(
                #         track_dir,
                #         "{}-{}-{}-{}.chroma".format(
                #             sample_idx, start_idx, end_idx, t.label
                #         ),
                #     )
                # )

                # chroma_mean = chroma_tensor.mean()
                # chroma_std = chroma_tensor.std()
                # chroma_means.append(chroma_mean)
                # chroma_stds.append(chroma_std)
                

                with open(label_fp, "a") as f:
                    f.write(
                        "{},{},{},{},{}\n".format(
                            t.track_id, sample_idx, start_idx, end_idx, t.label
                        )
                    )

    return (
        np.array(means).mean(),
        np.array(stds).mean(),
        np.array(chroma_means).mean(),
        np.array(chroma_stds).mean(),
        audio_len_seconds,
        class_weights,
    )


def train_test(args, tracks, subsample=True):
    analysed_tracks = []
    for t in tqdm(tracks):
        t = tracks[t]
        audio_path = t.audio_path
        if os.path.exists(audio_path):
            if args.dataset == "beatles":
                if t.key is None:
                    continue

                tonic = t.key.keys[0]
                if ":" in tonic:  # remove dorian, mixolydian, etc.
                    tonic = tonic[: tonic.find(":")]

                mode = estimate_mode(tonic, t.chords)
            else:
                tonic = t.salami_metadata()["tonic"][0]
                mode = estimate_mode(tonic, t.chords["majmin"])

            if mode is None:  # discard song if mode is unknown
                continue

            key = tonic + ":" + mode
            key = key_to_label(key)
            mode = mode_to_label(mode)

            t.tonic = tonic
            t.mode = mode
            t.key = key

            if args.task == "mode":
                t.label = t.mode
            elif args.task == "key":
                t.label = t.key
            else:
                raise NotImplementedError

            analysed_tracks.append(t)

    # np.random.shuffle(analysed_tracks)  # TODO
    train_l = int(len(analysed_tracks) * 0.8)
    train = analysed_tracks[:train_l]
    test = analysed_tracks[train_l:]
    if subsample:
        train = resample_tracks(args, train)
        test = resample_tracks(args, test)

    train_class_weights = np.zeros(args.n_classes)
    test_class_weights = np.zeros(args.n_classes)

    for t in train:
        train_class_weights[t.label] += 1
    for t in test:
        test_class_weights[t.label] += 1

    print("train_class_weights", train_class_weights)
    print("test_class_weights", test_class_weights)

    # shuffle train / test once
    np.random.shuffle(train)
    np.random.shuffle(test)
    return train, test


def write_statistics(mean, std, num_songs, stats_fp):
    with open(stats_fp, "w") as f:
        f.write("mean;std;num_songs\n")
        f.write(
            ";".join(
                [
                    str(mean),
                    str(std),
                    str(num_songs)
                ]
            )
        )
