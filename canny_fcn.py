import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import keras_model as km

from layers_fcn import (
    apply_noise, multiply, add, ActivityRegularization,
    GaussFilterLayer, GradientLayer, GradientDirection, SquareLayer,
    Conv1x1, ClipValues, NonMaximumSuppression, ThresholdLayer, dir_filters
)


def create_canny_model(thresh=0.0, use_gauss=True, gauss_ksize=5,
                       l1_reg=0.0, l2_reg=0.0, train_conv=False):
    img = tf.keras.Input(shape=[None, None, 1], name='image')
    grad_inp = img
    if use_gauss:
        gauss = GaussFilterLayer(
            kernel_size=gauss_ksize,
            trainable=train_conv,
            name='gauss'
        )(img)
        grad_inp = gauss

    grad = GradientLayer(
        trainable=train_conv,
        name='gradient'
    )(grad_inp)

    direction = GradientDirection(name='grad_direction')(grad)
    grad2 = SquareLayer(name='gradient2')(grad)
    magnitude = Conv1x1(
        kernel_initializer=[1, 1],
        activation=tf.sqrt,
        name='magnitude',
        trainable=train_conv
    )(grad2)

    suppressed_lst = []
    for d in dir_filters():
        # 1 where corresponds to current direction, else 0
        curr_dir = ClipValues(
            min_value=d['min_angle'],
            max_value=d['max_angle'],
            name='direction_%s' % d['name']
        )(direction)

        # M value, where maximum in corresponding direction
        curr_nms = NonMaximumSuppression(
            mask=d['filter'],
            name='nms_%s' % d['name']
        )(magnitude)

        # combine current direction
        curr_suppressed = multiply(
            [curr_nms, curr_dir],
            name='suppressed_dir_%s' % d['name']
        )
        suppressed_lst.append(curr_suppressed)

    # Combine all directions
    suppressed = add(suppressed_lst, name='suppressed')

    low_thresh = ThresholdLayer(
        name='low_threshold',
        bias_initializer=tf.constant_initializer(-thresh),
        trainable=True,
        activation='sigmoid'
    )(suppressed)

    # regularized = ActivityRegularization(l1=l1_reg, l2=l2_reg)(low_thresh)
    model = tf.keras.Model(inputs=img, outputs=low_thresh)
    return model


def ocv_canny(img, t):
    # img = img.astype(np.float32)
    # gauss = cv2.GaussianBlur(img, (5, 5), 1.4)
    ocv_edges = cv2.Canny(img, t, t, apertureSize=3, L2gradient=True)
    ocv_edges = ocv_edges.astype(np.float32) / 255.0
    return ocv_edges


def test_canny(path, t=200):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ocv_edges = ocv_canny(img, t=t)

    model = km.Model('temp_canny', model=create_canny_model(thresh=t / 255, use_gauss=False))
    fcn_edges = model.normalize_and_predict(img)
    fcn_edges_binary = km.binarize(fcn_edges, 0.5)

    edges_adf = km.adf(fcn_edges_binary, ocv_edges, 2)
    print('Wrong points: %d' % edges_adf.astype(bool).sum())

    f, ax = plt.subplots(2, 3)
    ax = ax.flatten()

    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(ocv_edges, cmap='gray')

    ax[3].imshow(fcn_edges, cmap='gray')
    ax[4].imshow(fcn_edges_binary, cmap='gray')
    ax[5].imshow(edges_adf, cmap='gray')

    f.show()


if __name__ == '__main__':
    img_path = '/home/az/data/edges2/HED-BSDS/test/100007.jpg'
    threshold = 200
    test_canny(path=img_path, t=threshold)
