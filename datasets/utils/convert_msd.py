import os
import pickle
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import subprocess
import torchaudio
from data.millionsongdataset import load_id2gt, load_id2path, default_loader
from datasets.utils.resample import convert_samplerate

def converter(path, conv_path, id2audio, sample_rate, max_length):
    items = []
    tracks_dict = defaultdict(list)
    # index = 0
    prev_index = None
    index = {}
    index_num = 0
    for idx, (clip_id, audio_path) in tqdm(enumerate(id2audio.items()), total=len(id2audio)):
        orig_fp = os.path.join(path, audio_path)
        orig_fp = os.path.splitext(orig_fp)[0] + ".mp3"
        target_fp = os.path.join(conv_path, os.path.splitext(audio_path)[0] + ".wav") 

        if os.path.exists(target_fp):
            continue

        if os.path.exists(orig_fp) and os.path.getsize(orig_fp) > 0:
            audio, sr = default_loader(orig_fp)
            if sr != sample_rate:
                tmp_fp = "/tmp/tmp.wav"
                convert_samplerate(orig_fp, tmp_fp, sample_rate)
                audio, sr_conv = default_loader(tmp_fp)
        
            audio = audio.mean(axis=0) # to mono
            audio = audio.reshape(1, -1) # [channels, samples]

            dp = Path(target_fp).parent
            if not os.path.exists(dp):
                os.makedirs(dp)

            # trim audio
            if audio.size(1) > max_length:
                audio = audio[:, 0:max_length]

            torchaudio.save(target_fp, audio, sample_rate)

        else:
            print("File not found or corrupted: {}".format(orig_fp))
    return items, tracks_dict


if __name__ == "__main__":
    data_input_dir = "/storage/jspijkervet"
    dir_name = "raw"
    sample_rate = 22050
    max_length = 30 * sample_rate # 30 seconds max len
    audio_dir = os.path.join(data_input_dir, "million_song_dataset", dir_name)
    conv_path = os.path.join(data_input_dir, "million_song_dataset", f"processed_{sample_rate}")
    msd_processed_annot = Path(
        data_input_dir, "million_song_dataset", "processed_annotations"
    )
    
    msd_to_7d = pickle.load(open(Path(msd_processed_annot) / "MSD_id_to_7D_id.pkl", "rb"))

    [audio_repr_paths, id2audio_repr_path] = load_id2path(
        Path(msd_processed_annot) / "index_msd.tsv", msd_to_7d
    )
    tracks_list, tracks_dict = converter(
        audio_dir, conv_path, id2audio_repr_path, sample_rate, max_length
    )


    exit(0)