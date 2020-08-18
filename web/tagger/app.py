from flask import Flask, jsonify, render_template, request
import os
import torchaudio
import torch
import numpy as np
import matplotlib.pyplot as plt
from data import get_dataset
from model import load_encoder
from scripts.datasets.resample import resample
from utils import parse_args, download_yt
from inference import save_taggram
import base64

tmp_input_file = "yt.mp3"

args = parse_args("./config/config.yaml")
args.world_size = 1
args.supervised = False
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data loaders
(
    train_loader,
    train_dataset,
    val_loader,
    val_dataset,
    test_loader,
    test_dataset,
) = get_dataset(args, pretrain=True, download=args.download)

# load pre-trained encoder
encoder = load_encoder(args, reload=True)
encoder.eval()
encoder = encoder.to(args.device)

model = None
if not args.supervised:
    finetuned_head = torch.nn.Sequential(
        torch.nn.Linear(args.n_features, args.n_features),
        torch.nn.ReLU(),
        torch.nn.Linear(args.n_features, args.n_classes)
    )

    finetuned_head.load_state_dict(
        torch.load(
            os.path.join(
                args.finetune_model_path,
                f"finetuner_checkpoint_{args.finetune_epoch_num}.pt",
            )
        )
    )
    finetuned_head = finetuned_head.to(args.device)



app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    video_link = request.form.get('youtubeLink')
    d = {}

    try:
        print("Downloading and converting YouTube video...")
        video_title, video_id = download_yt(video_link, tmp_input_file, args.sample_rate)
        conv_fn = f"{tmp_input_file}_{args.sample_rate}.wav"
        print("Resampling...")
        resample(tmp_input_file, conv_fn, args.sample_rate)

        yt_audio = train_dataset.get_audio(conv_fn)

        # to mono
        yt_audio = yt_audio.mean(axis=1).reshape(1, -1)
        yt_audio = torch.from_numpy(yt_audio)  # numpy to torch tensor

        # split into equally sized tensors of args.audio_length
        chunks = torch.split(yt_audio, args.audio_length, dim=1)
        chunks = chunks[
            :-1
        ]  # remove last one, since it's not a complete segment of audio_length

        print("Starting inference...")
        with torch.no_grad():
            taggram = []
            for idx, x in enumerate(chunks):
                x = x.to(args.device)

                # normalise
                if train_dataset.mean:
                    x = train_dataset.normalise_audio(x)

                # add batch dim
                x = x.unsqueeze(0)
                h = encoder(x)

                output = finetuned_head(h)
                output = torch.nn.functional.softmax(output, dim=1)
                output = output.squeeze(0)  # remove batch dim
                taggram.append(output.cpu().detach().numpy())
        
        print("Cleaning up mp3...")
        os.remove(tmp_input_file)
        os.remove(conv_fn)


        fn = "{}_taggram.png".format(base64.b64encode(video_title.encode("ascii")))
        taggram_fp = os.path.join(app.static_folder, "images", fn)
        save_taggram(yt_audio, video_title, args.sample_rate, args.audio_length, taggram, train_dataset.tags, taggram_fp)

        taggram = np.array(taggram)
        scores = {}
        for score, tag in zip(taggram.mean(axis=0), train_dataset.tags):
            scores[tag] = score
        
        scores = {k: v for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)}
        d["scores"] = [float(s) for s in scores.values()]
        d["tags"] = [str(t) for t in scores.keys()]
        d["image"] = fn
        d["video_title"] = video_title
        d["video_link"] = video_link
        d["video_id"] = video_id
        error = ""
    except Exception as e:
        print(e)
        error = e
        pass

    d["error"] = str(error)
    return jsonify(d)


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)