from past.builtins import xrange
import numpy as np
import scipy.misc as misc
import pandas as pa
import re
import os
import sys
import imageio
import torch

class DeepScoresProcessor():
    path = ""
    class_mappings = ""
    files = []
    images = []
    annotations = []
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, path, seed=42, split=0.2, min_nr=2, one_hot=True):
        """
        Initialize a file reader for the DeepScores classification data
        :param path: path to the dataset
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        """
        print("Initializing DeepScores Classification Batch Dataset Reader...")
        self.path = path
        self.seed = seed

        self.class_names = pa.read_csv(self.path+"/class_names.csv", header=None)

        config = open(self.path+"/config.txt", "r")
        config_str = config.read()
        self.tile_size = re.split('\)|,|\(', config_str)[4:6]

        self.tile_size[0] = int(self.tile_size[0])
        self.tile_size[1] = int(self.tile_size[1])

        self.seed = seed
        self.min_nr = min_nr
        self.split = split
        self.one_hot = one_hot

        # show image
        # from PIL import Image
        # im = Image.fromarray(self.images[234])
        # im.show()
        # print self.annotations[234]

    def read_images(self):
        for folder in os.listdir(self.path):
            if os.path.isdir(self.path +"/"+folder) and max(self.class_names[1].isin([folder])):
                    class_index = int(self.class_names[self.class_names[1] == folder][0])
                    self.load_class(folder,class_index)
                    print(folder + " loaded")

        # cast into arrays
        self.images = np.stack(self.images)
        self.annotations = np.stack(self.annotations)

        # extract test data
        test_indices = []
        train_indices = []
        print("splitting data: " + str(1 - self.split) + "-training " + str(self.split) + "-testing")
        for cla in np.unique(self.annotations):
            if sum(self.annotations == cla) < self.min_nr:
                print(
                "Less than " + str(self.min_nr) + " occurences - removing class " + self.class_names[1][cla])
            else:
                # do split
                cla_indices = np.where(self.annotations == cla)[0]
                np.random.shuffle(cla_indices)
                train_indices.append(cla_indices[0:int(len(cla_indices) * (1 - self.split))])
                test_indices.append(cla_indices[int(len(cla_indices) * (1 - self.split)):len(cla_indices)])

        train_indices = np.concatenate(train_indices)
        test_indices = np.concatenate(test_indices)


        self.test_images = self.images[test_indices]
        self.test_annotations = self.annotations[test_indices]

        self.images = self.images[train_indices]
        self.annotations = self.annotations[train_indices]

        # Shuffle the data
        perm = np.arange(self.images.shape[0])
        np.random.seed(self.seed)
        np.random.shuffle(perm)
        self.images = self.images[perm]
        self.annotations = self.annotations[perm]

        if sum(np.unique(self.annotations) != np.unique(self.test_annotations)) != 0:
            print("NOT THE SAME CLASSES IN TRAIN AND TEST - EXITING")
            sys.exit(1)

        self.nr_classes = max(self.test_annotations) + 1
        if self.one_hot:
            self.annotations = np.eye(self.nr_classes, dtype=np.uint8)[self.annotations]
            self.test_annotations = np.eye(self.nr_classes, dtype=np.uint8)[self.test_annotations]


        print(len(self.images))
        print(len(self.annotations))

        # hwc
        print(self.images[0].shape)
        # train_X = torch.from_numpy(self.images)
        # train_y = torch.from_numpy(self.annotations)
        # test_X = torch.from_numpy(self.test_images)
        # test_y = torch.from_numpy(self.test_annotations)


        # save tensors for training
        np.save("./datasets/vision/deepscores/train", self.images)
        np.save("./datasets/vision/deepscores/train_annotations", self.annotations)
        np.save("./datasets/vision/deepscores/test", self.test_images)
        np.save("./datasets/vision/deepscores/test_annotations", self.test_annotations)

        # self.__channels = True
        # self.images = np.array([self._transform(filename['image']) for filename in self.files])
        # self.__channels = False
        # self.annotations = np.array(
        #     [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
        # print (self.images.shape)
        # print (self.annotations.shape)

    def load_class(self, folder, class_index):
        # move trough images in folder
        for image in os.listdir(self.path +"/"+folder):
            self.load_image(folder, image, class_index)
        return None

    def load_image(self,folder,image, class_index):
        image = imageio.imread(self.path + "/" + folder + "/" + image)
        nr_y = image.shape[0] // self.tile_size[0]
        nr_x = image.shape[1] // self.tile_size[1]

        for x_i in xrange(0, nr_x):
            for y_i in xrange(0, nr_y):
                img = image[y_i*self.tile_size[0]:(y_i+1)*self.tile_size[0], x_i*self.tile_size[1]:(x_i+1)*self.tile_size[1]]
                self.images.append(img)
                self.annotations.append(class_index)
        return None


if __name__ == "__main__":
    data_reader = DeepScoresProcessor("./datasets/vision/deepscores")
    data_reader.read_images()

    #data_reader = Classification_BatchDataset("../Datasets/classification_data")