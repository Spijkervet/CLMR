import os
import numpy as np
import argparse
import torch
import multiprocessing
import torchaudio
from pathlib import Path
from tqdm import tqdm
from .resample import resample


def chunk(lst, n):
    return list(zip(*[iter(lst)] * n))


def process(split, data_input_dir, sample_rate, track_id, clip_id, segment):
    fp = train_dataset.id2audio_path[clip_id]
    src_path = os.path.join(audio_dir, fp)
    target_dir = os.path.join(data_input_dir, base_dir, "processed", split)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    target_fn = f"{track_id}-{clip_id}-{sample_rate}.wav"
    target_path = os.path.join(target_dir, target_fn)

    # resample to target sample rate
    if not os.path.exists(target_path):
        resample(src_path, target_path, sample_rate)


def process_dataset(split, dataset, data_input_dir, sample_rate):
    # get all tracks and convert them to |audio_length| segments

    print("Processing all tracks (this may take a while)")
    p = multiprocessing.Pool()
    for track_id, clip_id, segment, _, _ in dataset.index:
        if segment == 0:
            p.apply_async(
                process,
                args=[split, data_input_dir, sample_rate, track_id, clip_id, segment],
            )

    p.close()
    p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_input_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--audio_length", type=int, required=True)
    parser.add_argument("--sample_rate", type=int, required=True)
    parser.add_argument("--file_format", type=str, default="wav")
    args = parser.parse_args()

    args.domain = "audio"
    args.batch_size = 64
    args.workers = 0  # number of threads in CPU
    args.load_ram = False # do not load data into memory for processing
    args.nodes = 1

    from data import get_dataset

    # data loaders
    (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    ) = get_dataset(args, pretrain=False, download=False)

    base_dir = train_dataset.base_dir
    audio_dir = train_dataset.audio_dir

    process_dataset("train", train_dataset, args.data_input_dir, args.sample_rate)
    process_dataset("valid", val_dataset, args.data_input_dir, args.sample_rate)
    process_dataset("test", test_dataset, args.data_input_dir, args.sample_rate)
