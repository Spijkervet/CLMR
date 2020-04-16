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
from pydub import AudioSegment

sample_rate = 22050
MTT_DIR = "./datasets/audio/magnatagtune/raw"
AUDIO_DIR = f"./datasets/audio/magnatagtune/processed_{sample_rate}"


def process(raw_path, path, audio, npyfilepath):
    fp = os.path.join(MTT_DIR, path, audio)

    fn = audio.split(".")[0]

    index_name = "-".join(fn.split("-")[:-2])
    search = os.path.join(raw_path, path, index_name + "*")
    all_mp3 = glob(search)

    try:

        new_fp = str(Path(npyfilepath) / (path + "/" + index_name + "-0" + "-full.mp3"))
        if not os.path.exists(new_fp):
            cmd = ["cat", *all_mp3, ">", new_fp]
            os.system(" ".join(cmd))
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "panic",
                "-i",
                new_fp,
                "-ar",
                str(sample_rate),
                new_fp,
            ]
            os.system(" ".join(cmd))

            # y, sr = librosa.load(new_fp, sr=sample_rate)
            # np.save(new_fp, y)

            # os.remove(new_fp)
            # sf.write(new_fp, y, samplerate=sample_rate)

            # cmd = [
            #     "ffmpeg",
            #     "-y",
            #     *["-i " + a for a in all_mp3],
            #     # *["-map " + str(i) for i in range(len(all_mp3))],
            #     "-vcodec",
            #     "copy",
            #     "-shortest",
            #     f"-ar {sample_rate}",
            #     new_fp,
            # ]
            # cmd = ["sox", "--combine", "concatenate", *all_mp3, new_fp, "splice", "-q", "$(soxi −D {}),1"]

            # combined = AudioSegment.from_mp3(all_mp3[0])
            # combined = combined.set_frame_rate(sample_rate) # resample
            # for mp3 in all_mp3[1:]:
            #     audio = AudioSegment.from_mp3(mp3)
            #     audio = audio.set_frame_rate(sample_rate) # resample
            #     combined = combined.append(audio, crossfade=1000) # to avoid clipping / phase distortion

            # combined.export(new_fp, format="mp3")
            # print("Done", index_name)

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

    mydir = [path for path in os.listdir(rawfilepath) if path >= "0" and path <= "f"]
    for path in tqdm(mydir):
        # create directory with names '0' to 'f' if it doesn't already exist
        try:
            os.mkdir(Path(npyfilepath) / path)
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
                p.apply_async(process, [rawfilepath, path, audio, npyfilepath])
                # process(rawfilepath, path, audio, npyfilepath)

        p.close()
        p.join()


if __name__ == "__main__":
    # read audio signal and save to npy format
    save_audio_to_npy(MTT_DIR, AUDIO_DIR)
