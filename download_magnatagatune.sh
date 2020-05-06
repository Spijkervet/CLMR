CWD=$(pwd)

mkdir -p datasets/audio/magnatagatune
cd datasets/audio/magnatagatune

wget -nc http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001
wget -nc http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002
wget -nc http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003
wget -nc http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv
cat mp3.zip.* > mp3_all.zip
unzip mp3_all.zip

# put all unprocessed files in the ./raw folder
mkdir raw
mv 0 1 2 3 4 5 6 7 8 9 a b c d e f raw


cd $CWD

mkdir -p datasets/audio/magnatagatune/processed_annotations
cd datasets/audio/magnatagatune/processed_annotations
wget -nc https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/index_mtt.tsv
wget -nc https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/train_gt_mtt.tsv
wget -nc https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/val_gt_mtt.tsv
wget -nc https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/test_gt_mtt.tsv

cd $CWD
# process to desired samplerate
python -m datasets.utils.process_magnatag --dataset magnatagatune --sample_rate 22050
# concat samples
python -m datasets.utils.concat_magnatag --dataset magnatagatune
# process concat samples to desired samplerate
python -m datasets.utils.process_magnatag --dataset magnatagatune --sample_rate 22050 --from_concat
