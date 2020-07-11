import sys
import os
from glob import glob
from tqdm import tqdm
import subprocess
import re

def resample(source, target, sample_rate):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-i",
            source,
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            target,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    stdout, stderr = process.communicate()
    matches = re.search(
        r"Duration:\s{1}(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),",
        str(stdout),
        re.DOTALL,
    ).groupdict()

    # if float(matches["seconds"]) < 20:
    #     print("removed track")
    #     os.remove(target)
