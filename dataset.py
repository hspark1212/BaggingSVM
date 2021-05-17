import numpy as np
import math

from sklearn.datasets import make_circles, make_moons, make_blobs

import tensorflow as tf
from tensorflow.keras.utils import Sequence


def make_dataset(data_style, n_samples=60000, n_positives=3000):
    """ make dataset with positive and unbalbeled data
    :param data_style : str, default circles
    :param n_samples : int, the number of positive and unlabeled data
    :param n_positives : int, the number of positive data in n_samples

    :return x_data : nd_array [n_samples, 2], (x, y) coordinate
    :return y_data :  nd_array [n_samples,], labels for 0 -> unlbaled, 1 -> positive
    """
    if data_style == "circle":
        x_data, y_data = make_circles(n_samples=n_samples, noise=0.1, shuffle=True, factor=.65)
    elif data_style == "moons":
        x_data, y_data = make_moons(n_samples=n_samples, noise=0.1, shuffle=True)
    elif data_style == "blobs":
        x_data, y_data = make_blobs(n_samples=n_samples, centers=[[1, 5], [5, 1], [0, 0], [6, 6]])
    else:
        raise NameError("data_style should be 'circles, moons, blobs'")

    idx_p = np.where(y_data == 1)[0]
    final_idx_p = np.random.choice(idx_p, replace=False, size=n_positives)

    y_data[:] = 0
    y_data[final_idx_p] = 1

    return x_data, y_data


class DataloaderBagging(Sequence):
    """ Dataloader for baggingANN"""
    def __init__(self, x, y, bagging_size=1):
        """
        :param x: tensor [B, n], samples
        :param y: tensor [B,], labels
        :param bagging_size: multiple of positive examples
        """
        self.x = x
        self.u = x[y == 0]
        self.p = x[y == 1]
        self.bagging_size = bagging_size
        self.idx_u = tf.squeeze(tf.where(y == 0))

    def __len__(self):
        return math.ceil(len(self.u) / len(self.p))

    def __getitem__(self, idx):
        """
        :return self.p : tensor [B, n], positive data (same for every iter)
        :return self.b : tensor [B, n], randomly choice bootstrap in unlabeld data
        """
        shuffled_idx_u = tf.random.shuffle(self.idx_u)
        idx_b = shuffled_idx_u[:len(self.p) * self.bagging_size]  # bootstrap
        b = tf.gather(self.x, idx_b)

        return self.p, b


class Dataloader(Sequence):
    """dataloader for one epochs"""
    def __init__(self, x, y, batch_size):
        """
        shuffle data
        :param x: tensor [B, n], samples
        :param y: tensor [B,], labels
        :param batch_size: multiple of positive examples
        """
        idx = tf.range(len(x))
        shuffled_idx = tf.random.shuffle(idx)

        x = tf.gather(x, shuffled_idx)
        y = tf.gather(y, shuffled_idx)

        self.x = x
        self.y = y

        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __getitem__(self, idx):
        """
        :return
        self.p : tensor [B, n], shuffled samples
        self.b : tensor [B,], shuffled labels
        """
        start_idx = idx * self.batch_size
        final_idx = (idx + 1) * self.batch_size

        return self.x[start_idx:final_idx], self.y[start_idx:final_idx]
