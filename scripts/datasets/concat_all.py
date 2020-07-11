import os
import argparse
import errno
import numpy as np
import torch
from pathlib import Path
import multiprocessing
from glob import glob
from tqdm import tqdm
import subprocess



def process(raw_path, path, audio, output_dir):
    fp = os.path.join(MTT_DIR, path, audio)

    fn = audio.split(".")[0]

    index_name = "-".join(fn.split("-")[:-2])
    # print(index_name)

    search = os.path.join(raw_path, path, index_name + "*")
    all_mp3 = sorted(glob(search), key=lambda x: int(x.split("-")[-2]))

    try:
        new_fp = str(Path(output_dir) / (path + "/" + index_name + "-full.mp3"))
        if not os.path.exists(new_fp):
            cmd = ["cat", *all_mp3, ">", new_fp]
            os.system(" ".join(cmd))

    except Exception as e:
        print("Cannot save audio {} {}".format(audio, e))
        # pass
        


def process_all(rawfilepath, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mydir = [path for path in os.listdir(rawfilepath) if path >= "0" and path <= "f"]
    for path in tqdm(mydir):
        # create directory with names '0' to 'f' if it doesn't already exist
        try:
            os.mkdir(Path(output_dir) / path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        p = multiprocessing.Pool()
        audios = [
            audio
            for audio in os.listdir(Path(rawfilepath) / path)
            if audio.split(".")[-1] == "mp3"
        ]
        for audio in tqdm(audios):
            if "mp3" in audio:
                p.apply_async(process, [rawfilepath, path, audio, output_dir])
                # process(rawfilepath, path, audio, output_dir)

        p.close()
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_input_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample_rate", required=True)
    args = parser.parse_args()

    print(f"Concatenating {args.dataset}")
    MTT_DIR = os.path.join(args.data_input_dir, f"{args.dataset}/raw")
    AUDIO_DIR = os.path.join(args.data_input_dir, f"{args.dataset}/concat")

    process_all(MTT_DIR, AUDIO_DIR)
