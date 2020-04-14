import sys

sys.path.append("mirdata")
import os
import subprocess
import mirdata
from difflib import SequenceMatcher

orig_audio_dir = "$HOME/The Beatles/"
min_ratio = 0.7


def find_track(album, track_name):
    for root, dirs, files in os.walk(orig_audio_dir, topdown=True):
        album_dir = os.path.dirname(os.path.join(orig_audio_dir, root, files[0])).split(
            "/"
        )[-1]
        _album = album.split("-")[-1]
        album_confidence = SequenceMatcher(None, album_dir, _album)
        if album_confidence.ratio() >= min_ratio:
            # print(album_dir, album, album_confidence.ratio())
            for track in files:
                _track = os.path.splitext(track)[0]
                track_confidence = SequenceMatcher(
                    None, _track, track_name.replace("-", "").replace("CD", "")
                )
                if track_confidence.ratio() >= min_ratio:
                    return (
                        album_dir,
                        album_confidence.ratio(),
                        track,
                        track_confidence.ratio(),
                    )
    return None, None, None, None


if __name__ == "__main__":
    mirdata.beatles.download()
    data = mirdata.beatles.load()
    missing = []
    for k, v in data.items():
        audio_path = v._track_paths["audio"][0]
        _, album, track = audio_path.split("/")

        album = album.replace("_", " ")
        track = track.replace("_", " ")

        track_name = os.path.splitext(track)[0]
        found_album, ac, found_track, tc = find_track(album, track_name)

        if found_album is None:
            missing.append([album, track_name, "|", found_album, ac, found_track, tc])
        else:
            orig_fp = os.path.join(orig_audio_dir, found_album, found_track)
            if os.path.exists(orig_fp):
                new_fp = os.path.dirname(audio_path)
                if not os.path.exists(new_fp):
                    os.makedirs(new_fp)

                if not os.path.exists(audio_path):
                    subprocess.call(["ffmpeg", "-i", orig_fp, audio_path])

    print("Missing files:")
    for m in missing:
        print(m)
