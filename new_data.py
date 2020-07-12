import os
import numpy as np
from tqdm import tqdm
import warnings
import scipy.io.wavfile
import scipy


def load_tracks(sample_rate, index):
    audios = []
    for _, _, _, fp, _ in tqdm(index):
        audio, sr = process_wav(sample_rate, fp, False)
        audios.append(audio)
    return audios


def load_set(sample_rate, set_dirname, use_ulaw):
    ulaw_str = "_ulaw" if use_ulaw else ""
    file_names = [fn for fn in os.listdir(set_dirname) if fn.endswith(".wav")]
    full_sequences = []
    for fn in tqdm(file_names):
        audio, _ = process_wav(sample_rate, os.path.join(set_dirname, fn), use_ulaw)
        full_sequences.append(audio)

    return full_sequences


def read_wav(filename):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sr, audio = scipy.io.wavfile.read(filename)
    return audio, sr


def write_wav(filename, sample_rate, data):
    scipy.io.wavfile.write(filename, sample_rate, data)


def process_wav(desired_sample_rate, filename, use_ulaw):
    audio, sr = read_wav(filename)
    audio = ensure_mono(audio)
    audio = wav_to_float(audio)
    if use_ulaw:
        audio = ulaw(audio)
    # audio = ensure_sample_rate(desired_sample_rate, sr, audio)
    audio = audio.astype(np.float32)
    # audio = float_to_uint8(audio)
    return audio, sr


def ulaw(x, u=255):
    x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
    return x


def float_to_pcm(x):
    x *= np.iinfo("int16").max
    x = x.astype("int16")
    return x


def float_to_uint8(x):
    x += 1.0
    x /= 2.0
    uint8_max_value = np.iinfo("uint8").max
    x *= uint8_max_value
    x = x.astype("uint8")
    return x


def wav_to_float(x):
    try:
        max_value = np.iinfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    except:
        max_value = np.finfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    x = x.astype("float64", casting="safe")
    x -= min_value
    x /= (max_value - min_value) / 2.0
    x -= 1.0
    return x


def ulaw2lin(x, u=255.0):
    max_value = np.iinfo("uint8").max
    min_value = np.iinfo("uint8").min
    x = x.astype("float64", casting="safe")
    x -= min_value
    x /= (max_value - min_value) / 2.0
    x -= 1.0
    x = np.sign(x) * (1 / u) * (((1 + u) ** np.abs(x)) - 1)
    x = float_to_uint8(x)
    return x


def ensure_sample_rate(desired_sample_rate, file_sample_rate, mono_audio):
    if file_sample_rate != desired_sample_rate:
        mono_audio = scipy.signal.resample_poly(
            mono_audio, desired_sample_rate, file_sample_rate
        )
    return mono_audio


def ensure_mono(raw_audio):
    """
    Just use first channel.
    """
    if raw_audio.ndim == 2:
        raw_audio = raw_audio[:, 0]
    return raw_audio


def concat_tracks(sample_rate, dir, split, track_index):
    new_index = []
    concat_fps = []
    for track_id, v in tqdm(track_index.items()):
        out_file = os.path.join(dir, split, f"{track_id}-{sample_rate}.wav")

        # only process files that do not exist yet
        if not os.path.exists(out_file):
            for clips in v:
                clip_id, segment, fp, label = clips
                # only use 1 segment, it is already the whole 30s file
                if segment == 0:
                    concat_fps.append(fp)

            audios = []
            for fp in concat_fps:
                audio, sr = process_wav(sample_rate, fp, False)
                audios.append(audio)

            audios = np.concatenate(audios)
            audios = float_to_pcm(audios)
            write_wav(out_file, sample_rate, audios)
            concat_fps = []

        new_index.append([track_id, 0, 0, out_file, 0])
    return new_index


if __name__ == "__main__":
    load_set(22050, "./datasets/magnatagatune/processed/train", False)
