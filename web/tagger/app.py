from flask import Flask, jsonify, render_template, request
import os
import torchaudio
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")

from data import get_dataset
from model import load_encoder
from scripts.datasets.resample import resample
from utils import parse_args, download_yt, process_wav
from inference import save_taggram
import base64

tmp_input_file = "yt.mp3"

args = parse_args("./config/config.yaml")
args.world_size = 1
args.supervised = False
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tags = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird', 'country', 'metal', 'female voice', 'choral']
args.n_classes = len(tags)

# load pre-trained encoder
encoder = load_encoder(args, reload=True)
encoder.eval()
encoder = encoder.to(args.device)

model = None
if not args.supervised:
    finetuned_head = torch.nn.Sequential(
        torch.nn.Linear(args.n_features, args.n_classes),
    )

    finetuned_head.load_state_dict(
        torch.load(
            os.path.join(
                args.finetune_model_path,
                f"finetuner_checkpoint_{args.finetune_epoch_num}.pt",
            ),
            map_location=args.device
        )
    )
    finetuned_head = finetuned_head.to(args.device)



app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    video_link = request.form.get('youtubeLink')
    d = {}

    print("Downloading and converting YouTube video...")
    video_title, video_id = download_yt(video_link, tmp_input_file, args.sample_rate)
    conv_fn = f"{tmp_input_file}_{args.sample_rate}.wav"
    print("Resampling...")
    resample(tmp_input_file, conv_fn, args.sample_rate)

    yt_audio, _ = process_wav(args.sample_rate, conv_fn, False)
    yt_audio = yt_audio.reshape(1, -1)
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

    fn = "{}_taggram.png".format(base64.b64encode(video_title.encode("utf-8")))
    taggram_fp = os.path.join(image_path, fn)
    save_taggram(yt_audio, video_title, args.sample_rate, args.audio_length, taggram, tags, taggram_fp)

    taggram = np.array(taggram)
    scores = {}
    for score, tag in zip(taggram.mean(axis=0), tags):
        scores[tag] = score
    
    scores = {k: v for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)}
    d["scores"] = [float(s) for s in scores.values()]
    d["tags"] = [str(t) for t in scores.keys()]
    d["image"] = fn
    d["video_title"] = video_title
    d["video_link"] = video_link
    d["video_id"] = video_id
    error = ""

    d["error"] = str(error)
    return jsonify(d)


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")


if __name__ == "__main__":
    image_path =  os.path.join(app.static_folder, "images")
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    app.run()