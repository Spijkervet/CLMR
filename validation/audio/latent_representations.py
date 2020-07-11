import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from utils import tsne_to_json
# from tsnecuda import TSNE


def plot_tsne(args, embedding, labels, epoch, step, num_labels):
    fp = os.path.join(args.model_path, "tsne", "{}-{}.png".format(epoch, step))
    if not os.path.exists(os.path.dirname(fp)):
        os.makedirs(os.path.dirname(fp))

    figure = plt.figure(figsize=(8, 8), dpi=120)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=labels.ravel())

    labels = labels.squeeze()

    d = {"x": embedding[:, 0], "y": embedding[:, 1], "label": labels}
    df = pd.DataFrame(data=d)

    g = sns.scatterplot(
        x="x",
        y="y",
        hue="label",
        palette=sns.color_palette("hls", num_labels),
        legend="full",
        data=df,
        alpha=1,
    )

    box = g.get_position()
    g.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position

    # Put a legend to the right side
    g.legend(loc="center right", bbox_to_anchor=(1.25, 0.5), ncol=1)

    plt.axis("off")
    plt.savefig(fp, bbox_inches="tight")
    return figure


def tsne(features):
    embedding = TSNE().fit_transform(features)
    return embedding


def audio_latent_representations(
    args, dataset, model, epoch, global_step, writer, train, max_tracks=20, vis=False
):

    if max_tracks is None:
        max_tracks = len(dataset.tracks_dict)

    batch_size = 10 # 16 for 20480 samples, and a max.
    input_size = (args.batch_size, 1, args.audio_length)

    print("### Processing representations through TSNE ###")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.eval()
    with torch.no_grad():
        if args.model_name == "cpc":
            model = model.model # quick fix

        x = torch.zeros(input_size).to(args.device)
        latent_rep_size = model.get_latent_size(x)
        features = torch.zeros(max_tracks, batch_size, latent_rep_size).to(args.device)

        labels = torch.zeros(max_tracks, batch_size).to(args.device)

        idx = 0
        for _, track_idx in enumerate(tqdm(dataset.track_index, total=max_tracks)):
            if idx == max_tracks:
                break

            model_in = dataset.sample_audio_by_track_id(
                track_idx, batch_size=batch_size
            )

            if not torch.is_tensor(model_in):
                continue

            # for bidx, _ in enumerate(model_in):
            #     # audio = dataset.unnorm(model_in[bidx])
            #     audio = model_in[bidx]
            #     torchaudio.save(f"{idx}_{bidx}_{track_idx}_.wav", audio, dataset.sample_rate)

            model_in = model_in.to(args.device)
            
            if args.model_name == "cpc":
                z, c = model.get_latent_representations(model_in)
                h = c # context vector
            else:
                h, z = model.get_latent_representations(model_in)

            features[idx, :, :] = h.reshape((batch_size, -1))
            labels[idx, :] = int(track_idx)
            idx += 1

    features = features.reshape(features.size(0) * features.size(1), -1).cpu()
    labels = labels.reshape(-1, 1).cpu().numpy()

    embedding = tsne(features)
    figure = plot_tsne(args, embedding, labels, epoch, global_step, num_labels=max_tracks)

    is_train = "train" if train else "test"
    writer.add_figure(f"TSNE_{is_train}", figure, global_step=global_step)
    writer.flush()

    model.train()
