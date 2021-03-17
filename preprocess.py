
import argparse
from tqdm import tqdm
from datasets import get_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="magnatagatune")
    parser.add_argument("--dataset_dir", type=str, default="./data")
    args = parser.parse_args()

    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")

    for i in tqdm(range(len(train_dataset))):
        train_dataset.preprocess(i)

    for i in tqdm(range(len(valid_dataset))):
        valid_dataset.preprocess(i)

    for i in tqdm(range(len(test_dataset))):
        test_dataset.preprocess(i)