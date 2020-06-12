import torchaudio
import numpy as np

def tensor_to_audio(fn, t, sr):
    torchaudio.save(fn, t.cpu(), sample_rate=sr)


def write_audio_tb(args, train_loader, test_loader, writer, num_audio=5):
    idx = np.random.choice(len(train_loader.dataset), num_audio)
    for idx, r_idx in enumerate(idx):
        ((x_i, x_j), y, track_id) = train_loader.dataset[r_idx]

        if train_loader.dataset.mean is not None:
            x_i = train_loader.dataset.denormalise_audio(x_i)
            x_j = train_loader.dataset.denormalise_audio(x_j)
        writer.add_audio(
            f"audio/train-{idx}-{track_id}-xi",
            x_i,
            global_step=None,
            sample_rate=args.sample_rate,
            walltime=None,
        )
        writer.add_audio(
            f"audio/train-{idx}-{track_id}-xj",
            x_j,
            global_step=None,
            sample_rate=args.sample_rate,
            walltime=None,
        )

    idx = np.random.choice(len(test_loader.dataset), num_audio)
    for idx, r_idx in enumerate(idx):
        ((x_i, x_j), y, track_id) = test_loader.dataset[r_idx]
        if train_loader.dataset.mean is not None:
            x_i = train_loader.dataset.denormalise_audio(x_i)
            x_j = train_loader.dataset.denormalise_audio(x_j)
            
        writer.add_audio(
            f"audio/test-{idx}-{track_id}-xi",
            x_i,
            global_step=None,
            sample_rate=args.sample_rate,
            walltime=None,
        )
        writer.add_audio(
            f"audio/test-{idx}-{track_id}-xj",
            x_j,
            global_step=None,
            sample_rate=args.sample_rate,
            walltime=None,
        )

