import os
import glob
import numpy as np
import mirdata
import torchaudio
from tqdm import tqdm

# mirdata.billboard.download()
# mirdata.billboard.validate()
# tracks = mirdata.billboard.load()
tracks = list(glob.iglob('../datasets/fma_converted/fma_medium/**/*.mp3', recursive=True))
print(len(tracks))
means = []
stds = []
audio_len_seconds = 0
sample_rate = 16000
audio_length = 20480

for t in tqdm(tracks):
    audio_path = t
    # audio_path = tracks[t].audio_path
    # audio_path = os.path.join(
    #     os.path.dirname(audio_path), str(sample_rate) + os.path.splitext(audio_path)[1]
    # )
    # print(audio_path)
    if os.path.exists(audio_path):
        audio, sr = torchaudio.load(audio_path, normalization=False)
        audio = audio.mean(axis=0).reshape(1, -1)  # to mono

        assert (
            sr == sample_rate
        ), "Sample rate is not consistent throughout the dataset"

        # discard last part that is not a full 10ms
        ms = sample_rate / 100
        max_length = audio.size(1) // ms * ms
        audio_range = np.arange(ms, max_length - audio_length - 0, ms)

        if len(audio_range) == 0:
            print('removed {}'.format(audio_path))
            with open('log.txt', 'a') as f:
                f.write('removed {}\n'.format(audio_path))
            os.remove(audio_path)

        mean = audio.mean()
        std = audio.std()

        audio_len_seconds += audio.size(1) / sr
        # print(tracks[t].title, audio.size(1) / sr)
        means.append(mean)
        stds.append(std)

print("mean", np.array(means).mean())
print("std", np.array(stds).mean())
print("audio_len_seconds", audio_len_seconds)
print("num_songs", len(means))

with open("dataset_statistics.txt", "w") as f:
    f.write("mean,std,audio_len_seconds,num_songs\n")
    f.write(
        ",".join(
            [
                str(np.array(means).mean()),
                str(np.array(stds).mean()),
                str(audio_len_seconds),
                str(len(means)),
            ]
        )
    )
