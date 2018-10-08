import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from cv2 import (imread, cvtColor, COLOR_BGR2RGB, blur)

import keras_model as km


def suppress_dots(img):
    img[img > 0] = 1
    img_mean = blur(img.astype(np.float32), (3, 3))
    # 0.(1) mean a single point
    img_mean[img_mean > 0.12] = 1
    img_mean[img_mean < 0.12] = 0
    return (img * img_mean).astype(np.uint16)


def rect(w, h):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))


def img_grad(img):
    dx = np.zeros(img.shape[:-1], dtype=np.int16)
    dy = dx.copy()
    for i in range(3):
        dx_curr, dy_curr = cv2.spatialGradient(img[:, :, i])

        for (d, d_curr) in [(dx, dx_curr), (dy, dy_curr)]:
            idx = abs(d_curr) > abs(d)
            d[idx] = d_curr[idx]

    d_ver, d_hor = dx[1:, 1:], dy[1:, 1:]
    return d_ver, d_hor


def make_edges_image(gt, img, edges_type, morph=False):
    hor = abs(gt[1:] - gt[:-1])[:, 1:]
    ver = abs(gt[:, 1:] - gt[:, :-1])[1:]

    if morph:
        hor = cv2.morphologyEx(hor, cv2.MORPH_CLOSE, rect(4, 4))
        ver = cv2.morphologyEx(ver, cv2.MORPH_CLOSE, rect(4, 4))

    if edges_type == 'hor':
        d = suppress_dots(hor)
    elif edges_type == 'ver':
        d = suppress_dots(ver)
    elif edges_type == 'abs':
        d = np.expand_dims(hor + ver, 2)
    elif edges_type == 'grad':
        h = np.expand_dims(suppress_dots(hor), 2)
        v = np.expand_dims(suppress_dots(ver), 2)
        d = np.concatenate((h, v), axis=2)
    elif edges_type == 'dir_grad':
        hor = cv2.morphologyEx(hor, cv2.MORPH_DILATE, rect(2, 2))
        ver = cv2.morphologyEx(ver, cv2.MORPH_DILATE, rect(2, 2))

        v, h = img_grad(img)
        v = v * ver
        h = h * hor

        v[abs(v) < 15] = 0
        h[abs(h) < 15] = 0
        h = np.expand_dims(suppress_dots(h), 2)
        v = np.expand_dims(suppress_dots(v), 2)
        d = np.concatenate((h, v), axis=2)
    else:
        raise Exception('type not recognized')

    if edges_type != 'dir_grad':
        d[d > 0] = 1
    return d


def load_gt(path, img, edges_type, morph=False):
    gt = loadmat(path)['groundTruth']

    shape = img.shape[:-1]
    assert gt.shape[0] == 1

    n_channels = 2 if edges_type in ('grad', 'dir_grad') else 1
    gt_sum = np.zeros((shape[0] - 1, shape[1] - 1, n_channels), np.uint16)

    for i in range(gt.shape[1]):
        assert gt[0, i].shape == (1, 1)
        assert gt[0, i][0, 0].shape == ()

        curr_gt = gt[0, i][0, 0][0]
        if curr_gt.shape == (481, 321):
            curr_gt = curr_gt.T

        assert curr_gt.shape == shape

        # TODO: return several GT
        edges_type_internal = edges_type if edges_type != 'top3' else 'abs'
        gt_edges = make_edges_image(curr_gt, img, edges_type_internal, morph)
        gt_sum += gt_edges

    if edges_type == 'top3':
        gt_sum[gt_sum < 3] = 0
    min_v = -1 if edges_type in ('dir_grad', ) else 0
    gt_sum = np.clip(gt_sum, min_v, 1)
    return gt_sum


def load_BSD(root_dir, subdir, edges_type, morph=False):
    gt_dir  = os.path.join(root_dir, 'groundTruth', subdir)
    img_dir = os.path.join(root_dir, 'images', subdir)
    names = os.listdir(img_dir)

    n_channels = 2 if edges_type in ('grad', 'dir_grad') else 1
    images = np.zeros((len(names), 320, 480, 3))
    ground = np.zeros((len(names), 320, 480, n_channels))

    for idx, img_name in enumerate(tqdm(names)):
        gt_name = os.path.splitext(img_name)[0] + '.mat'
        img_path = os.path.join(img_dir, img_name)
        gt_path =  os.path.join(gt_dir, gt_name)
        if not os.path.exists(gt_path):
            raise Exception('GT for not found at %s!' % gt_path)

        img = imread(img_path)
        img = cvtColor(img, COLOR_BGR2RGB)

        if img.shape == (481, 321, 3):
            img = np.transpose(img, (1, 0, 2))
        assert img.shape == (321, 481, 3)

        gt = load_gt(gt_path, img, edges_type, morph)

        images[idx] = img[1:, 1:, :]
        ground[idx, :, :, :] = gt

    return images, ground


def main():
    src_dir = '/media/az/Data1/data/edges/BSR/BSDS500/data/'
    dst_dir = '/home/az/data/canny/data_top3'
    km.make_dir(dst_dir)

    for set in ('train', 'val', 'test'):
        img, gt = load_BSD(src_dir, subdir=set, edges_type='top3', morph=True)

        np.savez(os.path.join(dst_dir, '%s.npz' % set), img)
        np.savez(os.path.join(dst_dir, '%s_gt.npz' % set), gt)


if __name__ == '__main__':
    main()
