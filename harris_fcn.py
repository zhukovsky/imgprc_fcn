import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import keras_model as km

from layers_fcn import (
    GradientLayer, SquareLayer, MeanFilterLayer, Conv1x1,
    get_slice, NonMaximumSuppression
)

from utils import (show_images, load_image)


def create_harris_model(kernel_size=(5, 5), k=0.04, t=0.0, with_nms=True):
    img = tf.keras.Input(shape=(None, None, 1), name='img')
    n = np.prod(kernel_size)

    # [dI/dx, dI/dy]
    grad = GradientLayer(name='gradient')(img)

    img_x_tmp = get_slice(0, name='img_x')(grad)
    img_y_tmp = get_slice(1, name='img_y')(grad)

    img_x = Conv1x1(kernel_initializer=0.05, name='scale_x')(img_x_tmp)
    img_y = Conv1x1(kernel_initializer=0.05, name='scale_y')(img_y_tmp)

    img_x2 = SquareLayer(name='img_x2')(img_x)
    img_y2 = SquareLayer(name='img_y2')(img_y)
    img_xy = tf.keras.layers.multiply([img_x, img_y], name='img_xy')

    mean_x2 = MeanFilterLayer(kernel_size=kernel_size, name='mean_x2')(img_x2)
    mean_y2 = MeanFilterLayer(kernel_size=kernel_size, name='mean_y2')(img_y2)
    mean_xy = MeanFilterLayer(kernel_size=kernel_size, name='mean_xy')(img_xy)

    # trace^2
    pre_trace = tf.keras.layers.concatenate([mean_x2, mean_y2], name='pre_trace')

    trace = Conv1x1(
        kernel_initializer=[1, 1],
        name='trace'
    )(pre_trace)

    trace2 = SquareLayer(name='trace2')(trace)

    # Determinant
    m0 = tf.keras.layers.multiply([mean_x2, mean_y2], name='det_mult_0')
    m1 = SquareLayer(name='det_mult_1')(mean_xy)

    pre_det = tf.keras.layers.concatenate([m0, m1], name='pre_det')

    det = Conv1x1(
        kernel_initializer=[1, -1],
        name='det'
    )(pre_det)

    # Compute response
    pre_response = tf.keras.layers.concatenate([det, trace2], name='pre_response')

    response = Conv1x1(
        kernel_initializer=[1, -k],
        use_bias=True,
        bias_initializer=-t,
        name='response',
        activation='relu'
    )(pre_response)

    # NMS
    mask = np.ones((3, 3), dtype=np.bool)
    mask[1, 1] = False
    nms = NonMaximumSuppression(mask=mask, name='nms')(response)

    model = tf.keras.Model(
        inputs=img,
        outputs=nms if with_nms else response
    )
    model.compile(optimizer='sgd', loss='mse')
    print(model.summary())
    return model


def train():
    model = km.Model('temp_harris', model=create_harris_model())

    x_train = np.zeros((1, 100, 100, 1), dtype=np.float32)
    y_train = x_train.copy()
    _ = model.fit((x_train, y_train, x_train, y_train), epochs=1, batch_size=1)


def test_harris(path):
    img = load_image(path, is_float=True)
    corners_ocv = cv2.cornerHarris(img, 5, 3, 0.04, cv2.BORDER_CONSTANT)
    # corners_orig[corners_orig < 0.0] = 0.0

    model = km.Model(
        name='harris_fcn',
        model=create_harris_model(
            kernel_size=(5, 5),
            k=0.04,
            t=0.0,
            with_nms=False  # testing mode
        )
    )

    corners_fcn = model.predict(img)
    corners_adf = km.adf(corners_fcn, corners_ocv, 3)
    print(corners_adf.max())

    f, axes = plt.subplots(2, 2)
    show_images(f, axes, [img, corners_fcn, corners_ocv, corners_adf])


if __name__ == '__main__':
    path = '/home/az/data/dibco/text_small.png'
    test_harris(path)
    # train()
