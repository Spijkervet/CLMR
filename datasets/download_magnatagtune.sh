mkdir -p audio/magnatagtune
cd audio/magnatagtune

wget -nc http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001
wget -nc http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002
wget -nc http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003
wget -nc http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv
cat mp3.zip.* > mp3_all.zip
unzip mp3_all.zip