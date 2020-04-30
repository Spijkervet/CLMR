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

# concat samples
python -m datasets.utils.concat_magnatag
# process to desired samplerate
python -m datasets.utils.process_magnatag
