from glob import glob
import numpy as np

from scipy.ndimage import zoom  # 0: NearestNeighbor, 1: LInear, 3: Bicubic

import nibabel as nib


depth_mean = 5  # 9.478715


class InputData:
    def __init__(self, batch_size, num_class):
        train_images = glob("../data/TRAINING/case_*/*DWI*/*.nii")
        train_labels = glob("../data/TRAINING/*/*OT*/*.nii")

        self.batch_size = batch_size
        self.num_class = num_class

        assert len(train_images) > 0
        assert len(train_images) == len(train_labels)

        self.train_set = [(image, label) for image, label in zip(train_images, train_labels)]
        self.eval_set = []
        print('Number of TrainSet :', len(self.train_set))

        self.batches_per_epoch = len(self.train_set) // self.batch_size

    def split_rate(self, split_rate=0.1):
        num_evals = int(split_rate * len(self.train_set))
        self.train_set = self.train_set[:-num_evals]
        self.eval_set = self.train_set[-num_evals:]

    def split_kfolds(self, epoch_num, num_folds=10):
        self.num_folds = num_folds
        num_evals = len(self.train_set) // self.num_folds
        self.fold_n = epoch_num % self.num_folds

        self.eval_set = self.train_set[self.fold_n*num_evals:(self.fold_n + 1)*num_evals]
        # Remove eval_set from train_set
        [self.train_set.remove(data) for data in self.eval_set]

    def split_trainset(self, epoch_num, split_type):
        if split_type == 'kfolds':
            num_folds = 10
            if epoch_num % num_folds == 0:
                self.train_set = self.train_set + self.eval_set
                np.random.shuffle(self.train_set)
                self.split_kfolds(epoch_num, num_folds)
        elif split_type == 'rate':
            self.split_rate()
            np.random.shuffle(self.train_set)

        self.batches_per_epoch = len(self.train_set) // self.batch_size

    def resize_image(self, data, image, label):
        sizes = data.header.get_zooms()

        zoom_rate = sizes[-1] / depth_mean
        zoomed_image = zoom(image, (1, 1, zoom_rate), order=1)
        zoomed_label = zoom(label, (1, 1, zoom_rate), order=0)

        resized_image = zoomed_image[:, :, zoomed_image.shape[2]//2-2:zoomed_image.shape[2]//2+2]
        resized_label = zoomed_label[:, :, zoomed_image.shape[2]//2-2:zoomed_image.shape[2]//2+2]

        return resized_image, resized_label

    def generate_dataset(self, batch_num):
        batch_set = self.train_set[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]

        batch_image = []
        batch_label = []
        for image_path, label_path in batch_set:
            data = nib.load(image_path)
            image = np.array(data.get_data())
            label_raw = np.array(nib.load(label_path).get_data())

            image, label_raw = self.resize_image(data, image, label_raw)

            # Reshape Image and Label to make batches
            image = np.expand_dims(image, 4)

            label = np.zeros(list(label_raw.shape) + [self.num_class])
            for n in range(self.num_class):
                label[:, :, :, n] = label_raw == n
            # label = label_raw

            image = np.expand_dims(image, 0)
            label = np.expand_dims(label, 0)

            batch_image.append(image)
            batch_label.append(label)

        batch_image = np.concatenate(batch_image, axis=0)
        batch_label = np.concatenate(batch_label, axis=0)
        return batch_image, batch_label
