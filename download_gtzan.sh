mkdir -p datasets/audio/gtzan
cd datasets/audio/gtzan
wget -nc http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar -zxvf genres.tar.gz

mv genres raw # for processing later

mkdir annotations
cd annotations
wget -nc https://raw.githubusercontent.com/jordipons/sklearn-audio-transfer-learning/master/data/index/GTZAN/train_filtered.txt
wget -nc https://raw.githubusercontent.com/jordipons/sklearn-audio-transfer-learning/master/data/index/GTZAN/valid_filtered.txt
wget -nc https://raw.githubusercontent.com/jordipons/sklearn-audio-transfer-learning/master/data/index/GTZAN/test_filtered.txt


python -m datasets.utils.process_magnatag --dataset gtzan --sample_rate 22050