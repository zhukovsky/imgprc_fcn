import os
import numpy as np
import tensorflow as tf

from .utils import Scaler

K = tf.keras.backend


def weighted_binary_cross_entropy(y_true, y_pred):
    b = K.mean(y_true, axis=(1, 2, 3)) * 2

    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    pos_ce_mean = K.mean(     y_true  * K.log(    y_pred), axis=(1, 2, 3))
    neg_ce_mean = K.mean((1 - y_true) * K.log(1 - y_pred), axis=(1, 2, 3))

    ce = (1 - b) * pos_ce_mean + b * neg_ce_mean
    return K.mean(-ce, axis=-1)


class WeightedMSE:
    def __init__(self, b_mult=1.0):
        self.b_mult = b_mult
        self.__name__ = 'WeightedMSE'

    def __call__(self, y_true, y_pred):
        # y in [0, 1]

        # b = m * sum(edges) / n, weight of the background pixel
        b = K.mean(y_true, axis=(1, 2, 3)) * self.b_mult
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # background covering
        zero_idx = K.cast(K.equal(y_true, 0), K.floatx())
        m_neg = K.mean(K.square(y_pred) * zero_idx, axis=(1, 2, 3))

        # edges covering
        non_zero_idx = K.cast(K.not_equal(y_true, 0), K.floatx())
        m_pos = K.mean(K.square(y_pred - y_true) * non_zero_idx, axis=(1, 2, 3))

        m = (1 - b) * m_pos + b * m_neg
        return K.mean(m, axis=-1)


def pos_covering(y_true, y_pred):
    return K.mean(y_true * y_pred) / (K.mean(y_true * y_true) + K.epsilon())


def neg_covering(y_true, y_pred):
    return 1.0 - pos_covering(1.0 - y_true, y_pred)


def pos_covering_np(y_true, y_pred):
    # ideal: 1.0
    return np.mean(y_true * y_pred) / (np.mean(y_true * y_true))


def neg_covering_np(y_true, y_pred):
    return 1.0 - pos_covering_np(1.0 - y_true, y_pred)


def get_data(path, max_count=None, grayscale=False):
    x_trn = np.load(os.path.join(path, 'train.npz'))
    x_val = np.load(os.path.join(path, 'val.npz'))
    y_trn = np.load(os.path.join(path, 'train_gt.npz'))
    y_val = np.load(os.path.join(path, 'val_gt.npz'))

    x_trn = x_trn[x_trn.files[0]]
    x_val = x_val[x_val.files[0]]
    y_trn = y_trn[y_trn.files[0]]
    y_val = y_val[y_val.files[0]]

    if max_count and max_count > 0:
        x_trn = x_trn[:max_count]
        x_val = x_val[:max_count]
        y_trn = y_trn[:max_count]
        y_val = y_val[:max_count]

    if grayscale:
        x_trn = x_trn.mean(axis=3, keepdims=True)
        x_val = x_val.mean(axis=3, keepdims=True)

    x_scaler = Scaler()
    x_trn_scaled = x_scaler.fit(x_trn)
    x_val_scaled = x_scaler.transform(x_val)

    return (x_trn_scaled, y_trn, x_val_scaled, y_val), x_scaler
