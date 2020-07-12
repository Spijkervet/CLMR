import os
import numpy as np
from tqdm import tqdm
import warnings
import scipy.io.wavfile

def load_set(sample_rate, set_dirname, use_ulaw):
    ulaw_str = '_ulaw' if use_ulaw else ''
    
    file_names = [fn for fn in os.listdir(set_dirname) if fn.endswith('.wav')]
    full_sequences = []
    for fn in tqdm(file_names):
        sequence = process_wav(sample_rate, os.path.join(set_dirname, fn), use_ulaw)
        sequence = sequence.astype(np.float32)
        full_sequences.append(sequence)

    # np.save(f"processed_{sample_rate}_train", full_sequences)

    return full_sequences

def read_wav(filename):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sr, audio = scipy.io.wavfile.read(filename)
    return audio, sr

def process_wav(desired_sample_rate, filename, use_ulaw):
    audio, sr = read_wav(filename)
    audio = ensure_mono(audio)
    audio = wav_to_float(audio)
    if use_ulaw:
        audio = ulaw(audio)
    audio = ensure_sample_rate(desired_sample_rate, file_sample_rate, audio)
    audio = audio.astype(np.float32)
    # audio = float_to_uint8(audio)
    return audio


def ulaw(x, u=255):
    x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
    return x


def float_to_uint8(x):
    x += 1.
    x /= 2.
    uint8_max_value = np.iinfo('uint8').max
    x *= uint8_max_value
    x = x.astype('uint8')
    return x


def wav_to_float(x):
    try:
        max_value = np.iinfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    except:
        max_value = np.finfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    x = x.astype('float64', casting='safe')
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.
    return x


def ulaw2lin(x, u=255.):
    max_value = np.iinfo('uint8').max
    min_value = np.iinfo('uint8').min
    x = x.astype('float64', casting='safe')
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.
    x = np.sign(x) * (1 / u) * (((1 + u) ** np.abs(x)) - 1)
    x = float_to_uint8(x)
    return x

def ensure_sample_rate(desired_sample_rate, file_sample_rate, mono_audio):
    if file_sample_rate != desired_sample_rate:
        mono_audio = scipy.signal.resample_poly(mono_audio, desired_sample_rate, file_sample_rate)
    return mono_audio


def ensure_mono(raw_audio):
    """
    Just use first channel.
    """
    if raw_audio.ndim == 2:
        raw_audio = raw_audio[:, 0]
    return raw_audio


if __name__ == "__main__":
    load_set(22050, "./datasets/magnatagatune/processed/train", False)