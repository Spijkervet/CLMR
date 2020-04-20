import torch
import torch.utils.data as data
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
import torchvision.transforms
from glob import glob
import imageio
from tqdm import tqdm


class UniversalSymbolDataset(data.Dataset):
    def __init__(self, args, train=True, transform=None, target_transform=None):
        self.args = args
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform

        self.class_names = {}
        self.class_ids = {}
        classes = sorted(
            os.listdir(os.path.join(args.universal_symbol_processed, "training"))
        )
        num_classes = len(classes)
        for idx, c in enumerate(classes):
            self.class_names[idx] = c
            self.class_ids[c] = idx

        data = []
        targets = []
        if self.train:
            image_dir = "training"
        else:
            image_dir = "test"

        data_file = os.path.join(args.universal_symbol_processed, image_dir + "_data")
        targets_file = os.path.join(
            args.universal_symbol_processed, image_dir + "_targets"
        )

        if not os.path.exists(data_file + ".npy"):
            print("Creating data / annotations from dataset")
            images = glob(
                str(Path(args.universal_symbol_processed) / image_dir / "*" / "*.png")
            )
            for fp in tqdm(images):
                class_name = Path(fp).parent.stem
                class_id = self.class_ids[class_name]

                data.append(imageio.imread(fp))
                targets.append(class_id)

            self.data = np.stack(data)
            self.targets = np.stack(targets)

            # one hot
            self.targets = np.eye(num_classes, dtype=np.uint8)[self.targets]

            np.save(data_file, self.data)
            np.save(targets_file, self.targets)
        else:
            print(f"[{image_dir}] Loaded data / annotations from dataset")
            self.data = np.load(data_file + ".npy")
            self.targets = np.load(targets_file + ".npy")

        self.targets_dict = defaultdict(list)
        for x, y in zip(self.data, self.targets):
            class_id = y.argmax()  # one-hot
            self.targets_dict[class_id].append(x)

        self.sample_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = target.astype(np.float32)
        return img, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    def get_class_name(self, class_id):
        return self.class_names[class_id]

    def sample_from_class_id(self, class_id, batch_size=40):
        batch_size = min(batch_size, len(self.targets_dict[class_id]))
        batch = torch.zeros(
            batch_size,
            self.args.image_channels,
            self.args.image_height,
            self.args.image_width,
        )
        for idx in range(batch_size):
            img = Image.fromarray(self.targets_dict[class_id][idx])
            img = self.sample_transform(img)
            batch[idx, :] = img
        return batch
