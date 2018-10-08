import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import keras_model as km
from canny_fcn import create_canny_model
from edges_utils import (
    weighted_binary_cross_entropy, WeightedMSE,
    pos_covering, neg_covering, get_data
)

from layers_fcn import apply_noise


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    (x_trn, y_trn, x_val, y_val), scaler = get_data(
        path='/home/az/data/canny/data_abs',
        grayscale=True
    )

    kmod = km.Model(
        root_path='/home/az/data/canny/canny_conv_models',
        model=create_canny_model(
            thresh=0.1,
            use_gauss=True,
            gauss_ksize=7,
            gauss_sigma=2.0,
            # l1_reg=1e-6,
            train_conv=True
        )
    )

    kmod.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.03),
        # loss=weighted_binary_cross_entropy,
        loss=WeightedMSE(2.0),
        metrics=[pos_covering, neg_covering]
    )

    if True:
        kmod.model = apply_noise(kmod.model, ratio=0.01)

    kmod.fit(
        (x_trn, y_trn, x_val, y_val),
        epochs=300,
        batch_size=32
    )

    test_idx = 5
    x_pred = kmod.predict(x_trn[test_idx])

    f, axes = plt.subplots(3, 1)
    axes[0].imshow(np.squeeze(x_trn[test_idx]), cmap='gray')
    axes[1].imshow(np.squeeze(y_trn[test_idx]), cmap='gray')
    axes[2].imshow(np.squeeze(x_pred), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
