from glob import glob
import numpy as np

from scipy.ndimage import zoom  # 0: NearestNeighbor, 1: LInear, 3: Bicubic
from scipy.ndimage import label as seg_label

import nibabel as nib

import matplotlib.pyplot as plt


depth_mean = 5  # 9.478715


def search_largest_label(label):
    label_num, count = seg_label(label, )

    size = np.bincount(label_num.ravel())
    size[0] = 0
    largest_num = size.argmax()

    clump_mask = np.where(label_num == largest_num, 1, 0)

    return clump_mask


class InputData:
    def __init__(self, batch_size, num_class):
        train_images = glob("../data/TRAINING/case_*/*DWI*/*.nii")
        train_labels = glob("../data/TRAINING/*/*OT*/*.nii")
        train_MTTs = glob("../data/TRAINING/case_*/*MTT*/*.nii")

        self.batch_size = batch_size
        self.num_class = num_class

        assert len(train_images) > 0
        assert len(train_images) == len(train_labels)

        self.train_set = [(image, label, mtt) for image, label, mtt in zip(train_images, train_labels, train_MTTs)]
        self.eval_set = []
        print('Number of TrainSet :', len(self.train_set))

        self.batches_per_epoch = len(self.train_set) // self.batch_size
        self.eval_per_epoch = len(self.eval_set) // self.batch_size

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
        self.eval_per_epoch = len(self.eval_set) // self.batch_size

    def resize_image(self, data, image, label, resize=True):
        if resize:
            sizes = data.header.get_zooms()

            zoom_rate = sizes[-1] / depth_mean
            zoomed_image = zoom(image, (1, 1, zoom_rate), order=1)
            zoomed_label = zoom(label, (1, 1, zoom_rate), order=0)
        else:
            zoomed_image = image
            zoomed_label = label

        depth = zoomed_image.shape[2]
        resized_image = zoomed_image[:, :, depth//2-2:depth//2+2]
        resized_label = zoomed_label[:, :, depth//2-2:depth//2+2]

        return resized_image, resized_label

    def generate_dataset(self, batch_num, is_training=True):
        if is_training:
            batch_set = self.train_set[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
        else:
            batch_set = self.eval_set[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]

        batch_image = []
        batch_label = []

        step = 0
        for image_path, label_path, mtt_path in batch_set:
            data = nib.load(image_path)
            image = np.array(data.get_data())

            mtt = np.array(nib.load(mtt_path).get_data())
            mtt = np.where(mtt == 0, 0, 1)
            mtt = search_largest_label(mtt)
            image = image * mtt

            # image = (image - np.min(image)) / (np.max(image) - np.min(image))

            label = np.array(nib.load(label_path).get_data())
            # 여기부터 label이 3개가 된다.
            label = np.where(label == 1, 2, mtt)

            image, label = self.resize_image(data, image, label)

            # Reshape Image and Label to make batches
            image = np.expand_dims(image, 4)

            # label = np.zeros(list(label_raw.shape) + [self.num_class])
            # for n in range(self.num_class):
            #     label[:, :, :, n] = label_raw == n

            image = np.expand_dims(image, 0)
            label = np.expand_dims(label, 0)

            # if step == 0:
            #     plt.title('min: %s, max: %s' % (str(image.min()), str(image.max())))
            #     plt.imshow(image[0, :, :, 2, 0])
            #     plt.show()
            # step += 1

            batch_image.append(image)
            batch_label.append(label)

        batch_image = np.concatenate(batch_image, axis=0)
        batch_label = np.concatenate(batch_label, axis=0)
        return batch_image, batch_label
