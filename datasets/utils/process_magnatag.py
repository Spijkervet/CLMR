import os
import errno
import numpy as np
import argparse
from pathlib import Path
import multiprocessing
from glob import glob
from tqdm import tqdm



def process(raw_path, path, audio, npyfilepath):
    fp = os.path.join(MTT_DIR, path, audio)

    fn = os.path.splitext(audio)[0]

    index_name = "-".join(fn.split("-")[:-2])
    search = os.path.join(raw_path, path, index_name + "*")
    all_mp3 = glob(search)
    
    try:
        new_fp = str(Path(npyfilepath) / (path + "/" + fn + "." + file_format))
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
        # print(cmd)
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
            print(e)
            pass

        p = multiprocessing.Pool()
        audios = [
            audio
            for audio in os.listdir(Path(rawfilepath) / path)
            if audio.split(".")[-1] == "mp3" or audio.split(".")[-1] == "wav"
        ]
        for audio in tqdm(audios):
            p.apply_async(process, [rawfilepath, path, audio, npyfilepath])
            # process(rawfilepath, path, audio, npyfilepath)

        p.close()
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample_rate", required=True)
    parser.add_argument("--file_format", default="wav")
    parser.add_argument("--from_concat", action="store_true")
    args = parser.parse_args()
    
    sample_rate = args.sample_rate
    file_format = args.file_format
    if args.from_concat:
        MTT_DIR = f"./datasets/audio/{args.dataset}/concat_16000" # concat default SR is 16000
        AUDIO_DIR = f"./datasets/audio/{args.dataset}/processed_concat_{sample_rate}_wav"
    else:
        MTT_DIR = f"./datasets/audio/{args.dataset}/raw"
        AUDIO_DIR = f"./datasets/audio/{args.dataset}/processed_{sample_rate}_wav"
    
    # read audio signal and save to npy format
    process_all(MTT_DIR, AUDIO_DIR)
