import os
import errno
import numpy as np
import torch
import torchaudio
import librosa
from pathlib import Path
import multiprocessing
from glob import glob
from tqdm import tqdm
import soundfile as sf

sample_rate = 16000
SOURCE_DIR = "./datasets/audio/fma/fma_medium"  # /processed
TARGET_DIR = f"./datasets/audio/fma/fma_medium_{sample_rate}"


def process(raw_path, path, audio, target_fp):
    fp = os.path.join(SOURCE_DIR, path, audio)

    fn = audio.split(".")[0]

    # index_name = "-".join(fn.split("-")[:-2])
    # search = os.path.join(raw_path, path, index_name + "*")
    # all_mp3 = glob(search)

    try:
        new_fp = str(Path(target_fp) / (path + "/" + fn + ".mp3"))
        # resample
        if not os.path.exists(new_fp):
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
            # print(cmd)
            os.system(" ".join(cmd))

    except Exception as e:
        print("Cannot save audio {} {}".format(audio, e))
        pass


def save_audio_to_npy(rawfilepath, npyfilepath):
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

    dirs = [path for path in os.listdir(rawfilepath)]
    for path in tqdm(dirs):
        # create directory with names '0' to 'f' if it doesn't already exist
        try:
            os.mkdir(Path(npyfilepath) / path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        p = multiprocessing.Pool()
        audio_path = Path(rawfilepath) / path
        if not os.path.isdir(audio_path):
            continue

        audios = [
            audio for audio in os.listdir(audio_path) if audio.split(".")[-1] == "mp3"
        ]
        for audio in tqdm(audios):
            if "mp3" in audio:
                p.apply_async(process, [rawfilepath, path, audio, npyfilepath])
                # process(rawfilepath, path, audio, npyfilepath)

        p.close()
        p.join()


if __name__ == "__main__":
    # read audio signal and save to npy format
    save_audio_to_npy(SOURCE_DIR, TARGET_DIR)
