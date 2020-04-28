mkdir -p datasets/audio/fma
cd datasets/audio/fma
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
unzip fma_metadata.zip
rm fma_metadata.zip

wget https://os.unil.cloud.switch.ch/fma/fma_medium.zip
unzip fma_medium.zip
# rm fma_small.zip
