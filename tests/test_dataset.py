import unittest
import pytest
from clmr.datasets import (
    get_dataset,
    AUDIO,
    LIBRISPEECH,
    GTZAN,
    MAGNATAGATUNE,
    MillionSongDataset,
)


class TestAudioSet(unittest.TestCase):

    datasets = {
        "librispeech": LIBRISPEECH,
        "gtzan": GTZAN,
        "magnatagatune": MAGNATAGATUNE,
        "msd": MillionSongDataset,
        "audio": AUDIO,
    }

    def test_dataset_names(self):
        for dataset_name, dataset_type in self.datasets.items():
            with pytest.raises(RuntimeError):
                _ = get_dataset(
                    dataset_name, "./data/audio", subset="train", download=False
                )

    def test_custom_audio_dataset(self):
        audio_dataset = get_dataset(
            "audio", "./tests/data/audioset", subset="train", download=False
        )
        assert type(audio_dataset) == AUDIO
        assert len(audio_dataset) == 1
