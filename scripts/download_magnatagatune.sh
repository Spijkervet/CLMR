# Download script for MagnaTagATune
# Executed in ./data/audio/magnatagatune.py

ROOT_DIR=$1
SAMPLE_RATE=$2
CWD=$(pwd)

mkdir -p $ROOT_DIR/magnatagatune
cd $ROOT_DIR/magnatagatune

wget -nc http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001
wget -nc http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002
wget -nc http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003
wget -nc http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv
rm -f mp3_all.zip
cat mp3.zip.* > mp3_all.zip

echo "Unzipping files"
unzip mp3_all.zip

# put all unprocessed files in the ./raw folder
mkdir raw
mv 0 1 2 3 4 5 6 7 8 9 a b c d e f raw


echo "Fetching annotations"
cd $CWD
mkdir -p $ROOT_DIR/magnatagatune/processed_annotations
cd $ROOT_DIR/magnatagatune/processed_annotations
wget -nc https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/index_mtt.tsv
wget -nc https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/train_gt_mtt.tsv
wget -nc https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/val_gt_mtt.tsv
wget -nc https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/test_gt_mtt.tsv
wget -nc https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/output_labels_mtt.txt

cd $CWD
echo "Pre-processing raw audio to desired sample rate"
python -m scripts.datasets.preprocess_dataset --data_input_dir $ROOT_DIR --dataset magnatagatune --sample_rate $SAMPLE_RATE --audio_length 59049
