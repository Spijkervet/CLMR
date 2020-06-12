mkdir -p datasets/audio/msd/processed_annotations
cd datasets/audio/msd/processed_annotations
wget -nc https://github.com/jordipons/musicnn-training/raw/master/data/index/msd/train_gt_msd.tsv
wget -nc https://github.com/jordipons/musicnn-training/raw/master/data/index/msd/val_gt_msd.tsv
wget -nc https://github.com/jordipons/musicnn-training/raw/master/data/index/msd/test_gt_msd.tsv
wget -nc https://github.com/jordipons/musicnn-training/raw/master/data/index/msd/output_labels_msd.txt
wget -nc https://github.com/jordipons/musicnn-training/raw/master/data/index/msd/index_msd.tsv
wget -nc https://github.com/jongpillee/music_dataset_split/raw/master/MSD_split/MSD_id_to_7D_id.pkl