#!/bin/sh
. ~/miniconda3/etc/profile.d/conda.sh
conda activate clmr

# python linear_evaluation.py --dataset magnatagatune --model_path /storage/jspijkervet/logs/25 --epoch_num 2140 --logistic_epochs 500 --logistic_lr 0.0001 --perc_train_data 0.5

# python linear_evaluation.py --dataset magnatagatune --model_path /storage/jspijkervet/logs/25 --epoch_num 2140 --logistic_epochs 500 --logistic_lr 0.0001 --perc_train_data 0.2

python linear_evaluation.py --dataset magnatagatune --model_path /storage/jspijkervet/logs/25 --epoch_num 2140 --logistic_epochs 5000 --logistic_lr 0.0001 --perc_train_data 0.1

python linear_evaluation.py --dataset magnatagatune --model_path /storage/jspijkervet/logs/25 --epoch_num 2140 --logistic_epochs 5000 --logistic_lr 0.0001 --perc_train_data 0.05

python linear_evaluation.py --dataset magnatagatune --model_path /storage/jspijkervet/logs/25 --epoch_num 2140 --logistic_epochs 5000 --logistic_lr 0.0001 --perc_train_data 0.02

python linear_evaluation.py --dataset magnatagatune --model_path /storage/jspijkervet/logs/25 --epoch_num 2140 --logistic_epochs 5000 --logistic_lr 0.0001 --perc_train_data 0.01
