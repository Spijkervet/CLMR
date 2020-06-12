import os
import errno
import numpy as np
import argparse
from pathlib import Path
import multiprocessing
from glob import glob
from tqdm import tqdm
import torch
import torchaudio

import soundfile as sf

def process(raw_path, path, audio, npyfilepath):
    fp = audio
    # index_name = "-".join(fn.split("-")[:-2])
    # search = os.path.join(raw_path, path, index_name + "*")
    # all_mp3 = glob(search)

    audio = audio.replace("raw", Path(npyfilepath).stem)
    dp = Path(audio).parent
    if not os.path.exists(dp):
        os.makedirs(dp)

    fn = os.path.splitext(audio)[0]
    try:
        new_fp = fn + "." + file_format
        # resample
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "panic",
            "-i",
            fp,
            "-ar",
            str(sample_rate),
            new_fp,
        ]
        # print(" ".join(cmd))
        os.system(" ".join(cmd))

    except Exception as e:
        print("Cannot save audio {} {}".format(audio, e))
        pass
    

def process_all(rawfilepath, npyfilepath):
    """ Save audio signal with sr=sample_rate to npy file
    Args :
        rawfilepath : path to the MTT audio files
        npyfilepath : path to save the numpy array audio signal
    Return :
        None
    """

    # make directory if not existing
    if not os.path.exists(npyfilepath):
        os.makedirs(npyfilepath)

    mydir = [path for path in os.listdir(rawfilepath)]
    for path in tqdm(mydir):
        raw_dir = Path(rawfilepath) / path
        print(raw_dir)
        if os.path.isfile(raw_dir):
            continue

        try:
            os.mkdir(Path(npyfilepath) / path)
        except OSError as e:
            pass

        p = multiprocessing.Pool()
        path = os.path.join(rawfilepath, path)
        audios = glob(f"{path}/**/*.mp3")
        for audio in tqdm(audios):
            p.apply_async(process, [rawfilepath, path, audio, npyfilepath])
            process(rawfilepath, path, audio, npyfilepath)

        p.close()
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-input-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample_rate", required=True)
    parser.add_argument("--file_format", default="wav")
    parser.add_argument("--from_concat", action="store_true")
    args = parser.parse_args()
    
    sample_rate = args.sample_rate
    file_format = args.file_format
    if args.from_concat:
        MTT_DIR = os.path.join(args.data_input_dir, f"{args.dataset}/concat_16000") # concat default SR is 16000
        AUDIO_DIR = os.path.join(args.data_iput_dir, f"{args.dataset}/processed_concat_{sample_rate}_{file_format}")
    else:
        MTT_DIR = os.path.join(args.data_input_dir, f"{args.dataset}/raw")
        AUDIO_DIR = os.path.join(args.data_input_dir, f"{args.dataset}/processed_segments_{sample_rate}_{file_format}")
    
    # read audio signal and save to npy format
    process_all(MTT_DIR, AUDIO_DIR)
