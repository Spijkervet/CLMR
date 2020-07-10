import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import numpy as np


def plot_tsne(args, embedding, labels, epoch, step, num_labels):
    fp = os.path.join(args.out_dir, "tsne", "{}-{}.png".format(epoch, step))
    if not os.path.exists(os.path.dirname(fp)):
        os.makedirs(os.path.dirname(fp))

    figure = plt.figure(figsize=(8, 8), dpi=120)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=labels.ravel())

    labels = labels.squeeze()

    d = {"x": embedding[:, 0], "y": embedding[:, 1], "label": labels}
    df = pd.DataFrame(data=d)

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
    g.legend(loc="center right", bbox_to_anchor=(1.5, 0.5), ncol=1)

    plt.axis("off")
    plt.savefig(fp, bbox_inches="tight")
    return figure


def tsne(features):
    embedding = TSNE().fit_transform(features)
    return embedding


def vision_latent_representations(
    args, dataset, model, optimizer, epoch, step, global_step, writer, train
):

    max_classes = 20
    batch_size = 20
    input_size = (
        args.batch_size,
        args.image_channels,
        args.image_height,
        args.image_width,
    )

    model.eval()
    with torch.no_grad():
        latent_rep_size = model.get_latent_size(input_size)
        features = torch.zeros(max_classes, batch_size, latent_rep_size).to(args.device)

        images = torch.zeros(max_classes, batch_size, args.image_channels, args.image_height, args.image_width).to(args.device)
        labels = torch.zeros(max_classes, batch_size).to(args.device)
        idx = 0
        for _, class_id in enumerate(dataset.targets_dict):

            if idx == max_classes:
                break

            model_in = dataset.sample_from_class_id(class_id, batch_size=batch_size)
            model_in = model_in.to(args.device)

            if model_in.shape[0] < batch_size:
                continue

            h, z = model.get_latent_representations(model_in)

            features[idx, :, :] = z.reshape((batch_size, -1))

            images[idx, :] = model_in 
            labels[idx, :] = int(class_id)
            idx += 1
    
    # squeeze classes / batch together
    features = features.reshape(features.size(0) * features.size(1), -1).cpu()
    images = images.reshape(-1, args.image_channels, args.image_height, args.image_width)
    labels = labels.reshape(-1, 1)

    # get label names
    labels_names = []
    for lb in labels.cpu().numpy():
        labels_names.append([f"{class_id}: {dataset.get_class_name(class_id)}" for class_id in lb])
    labels_names = np.array(labels_names)

    embedding = tsne(features)
    figure = plot_tsne(
        args, embedding, labels_names, epoch, step, num_labels=max_classes
    )

    is_train = "train" if train else "test"
    writer.add_figure(f"TSNE_{is_train}", figure, global_step=global_step)
    writer.flush()

    out_dir = os.path.join(args.out_dir, "tsne")

    if train and epoch % 20 == 0:
        # writer.add_embedding(features, label_img=images, metadata=labels_names, global_step=global_step)
        torch.save(
            features, os.path.join(out_dir, "features-{}-{}.pt".format(epoch, step))
        )
        torch.save(labels, os.path.join(out_dir, "labels-{}-{}.pt".format(epoch, step)))
        writer.flush()

    model.train()
