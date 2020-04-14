import os
import sys

from pathlib import Path
import subprocess
import multiprocessing
import torchaudio
from tqdm import tqdm
from glob import glob

from utils.resample import convert_samplerate


def resampler(audio_path, target_sr):
    src = Path(audio_path)
    target = os.path.join(src.parent, src.stem) + f"_{target_sr}.mp3"

    if not os.path.exists(target):
        convert_samplerate(src, target, target_sr)


if __name__ == "__main__":

    target_sr = 16000

    files = list(glob("audio/fma_small/**/*.mp3"))

    p = multiprocessing.Pool()
    for f in tqdm(files):
        p.apply_async(resampler, [f, target_sr]) # async
        # resampler(f, target_sr) # sync
        
    p.close()
    p.join()
