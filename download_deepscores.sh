mkdir -p datasets/vision/deepscores
cd datasets/vision/deepscores
wget -nc https://repository.cloudlab.zhaw.ch/artifactory/deepscores/classification/DeepScores2017_classification.zip
unzip DeepScores2017_classification.zip
wget -nc https://tuggeluk.github.io/class_names/class_names.csv