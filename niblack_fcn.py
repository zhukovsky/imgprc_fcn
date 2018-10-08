import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import keras_model as km
from layers_fcn import (SquareLayer, MeanFilterLayer, Conv1x1)


def create_niblack_model(kernel_size=(10, 10), k=0.2, a=0.0):
    n = np.prod(kernel_size)
    img = tf.keras.Input(shape=(None, None, 1), name='img')

    # sum(x)^2
    mean = MeanFilterLayer(kernel_size, name='mean')(img)
    mean_2_x = SquareLayer(name='mean_2_x')(mean)

    # sum(x^2)
    img2 = SquareLayer(name='img2')(img)
    mean_x_2 = MeanFilterLayer(kernel_size, name='mean2')(img2)

    # [sum(x^2), sum(x) ^ 2]
    pre_std = tf.keras.layers.concatenate([mean_x_2, mean_2_x], name='pre_std')

    # unbiased sample variance
    # [sum(x^2) - sum(x)^2 / n] / (n-1) == std(x)
    std = Conv1x1(
        kernel_initializer=[1/(n-1), -1/n/(n-1)],
        activation=tf.sqrt,
        name='std'
    )(pre_std)

    # thresh = mean(x) + k *  std(x) + a
    pre_thresh = tf.keras.layers.concatenate([mean, std], name='pre_thresh')

    thresh = Conv1x1(
        kernel_initializer=[1/n, k],
        use_bias=True,
        bias_initializer=a,
        name='thresh',
        trainable=True
    )(pre_thresh)

    # binarized = sigma(img - thresh)
    pre_binarize = tf.keras.layers.concatenate(
        [img, thresh], name='pre_binarize')

    binarized = Conv1x1(
        kernel_initializer=[1, -1],
        name='binarize',
        activation='sigmoid'
    )(pre_binarize)

    model = tf.keras.Model(inputs=img, outputs=binarized)
    model.compile(optimizer='sgd', loss='mse')
    print(model.summary())
    return model


def train():
    model = km.Model('niblack', model=create_niblack_model(kernel_size=(10, 10)))

    x_train = np.zeros((1, 100, 100, 1), dtype=np.float32)
    y_train = x_train.copy()
    _ = model.fit((x_train, y_train, x_train, y_train), epochs=1, batch_size=1)


def ocv_niblack(img, w, k, a=0.0):
    img2 = np.square(img)

    ave  = cv2.blur(img,  (w, w), borderType=cv2.BORDER_CONSTANT)
    ave2 = cv2.blur(img2, (w, w), borderType=cv2.BORDER_CONSTANT)

    n = np.multiply(*img.shape)
    std = np.sqrt(ave2 * n / (n-1) - (ave ** 2) * n / (n-1))

    t = ave + k * std + a
    binary = np.zeros(img.shape)
    binary[img >= t] = 1.0
    return binary


def test_niblack():
    path = '/home/az/data/dibco/text_small.png'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(float) / 255.0

    niblack_ocv = ocv_niblack(img, 35, -0.7)

    model = km.Model(
        'temp_niblack',
        model=create_niblack_model(kernel_size=(35, 35), k=-0.7)
    )

    niblack_fcn = model.predict(img)
    niblack_fcn_bin = km.binarize(niblack_fcn, 0.5)

    f, ax = plt.subplots(2, 2)
    ax = ax.flatten()

    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(niblack_fcn, cmap='gray')

    ax[2].imshow(niblack_ocv, cmap='gray')
    ax[3].imshow(niblack_fcn_bin, cmap='gray')
    f.show()

    print(km.adf(niblack_ocv, niblack_fcn_bin).max())


if __name__ == '__main__':
    train()
