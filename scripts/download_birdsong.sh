# Download script for birdsong

ROOT_DIR=$1
SAMPLE_RATE=$2
CWD=$(pwd)

mkdir -p $ROOT_DIR/birdsong
cd $ROOT_DIR/birdsong

# put all unprocessed files in the ./raw folder
mv train_audio raw

cd $CWD
echo "Pre-processing raw audio to desired sample rate"
python -m scripts.datasets.preprocess_dataset --data_input_dir $ROOT_DIR --dataset birdsong --sample_rate $SAMPLE_RATE --audio_length 59049
