import cv2, os, json, itertools, h5py, time
import numpy as np
from jsmin import jsmin
import matplotlib.pyplot as plt


def show_images(f, axes, imgs):
    axes = axes.flatten()
    assert len(imgs) <= len(axes)

    for i in range(len(imgs)):
        axes[i].imshow(imgs[i], cmap='gray')

    f.show()


def load_image(path, is_float=False):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if is_float:
        img = img.astype(np.float32) / 255

    return img


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif not os.path.isdir(dir_path):
        raise Exception("Couldn't create a directory %s" % dir_path)


def dump_json(data, path):
    with open(path, 'w') as fout:
        json.dump(data, fout)


def load_json(path):
    with open(path, 'r') as fin:
        data = json.loads(jsmin(fin.read()))
    return data


def load_list(path):
    dir_path = os.path.dirname(path)

    with open(path, 'r') as fin:
        # read pure paths
        lines = [x.strip() for x in fin.readlines()]
        # drop missed lines
        lines = [x for x in lines if len(x) > 0]
        # drop comments
        lines = [x for x in lines if not x.startswith(';')]
        # includes
        lines = [load_list(os.path.join(dir_path, x[9:]), True)
                 if x.startswith('#include') else [x] for x in lines]
        # flat
        lines = list(itertools.chain.from_iterable(lines))
        # global paths
        lines = [os.path.join(dir_path, x) for x in lines]
        lines = [os.path.normpath(x) for x in lines]
    return lines


def dump_hdf5(path, data, name):
    np_array = np.asarray(data)
    with h5py.File(path, 'w') as f:
        f.create_dataset(name, data=np_array)


def load_hdf5(path, name):
    with h5py.File(path, 'r') as f:
        return f[name][:]


class Scaler:
    def __init__(self):
        self.mean, self.std = 0.0, 1.0

    def fit(self, x):
        self.mean, self.std = x.mean(), x.std()
        if self.std == 0:
            self.std = 1
        # self.std[self.std == 0] = 1
        return self.transform(x)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean

    def write(self, path):
        data_dict = dict(name='km_scaler', mean=self.mean, std=self.std)
        dump_json(data_dict, path)

    def read(self, path):
        data_dict = load_json(path)
        if not data_dict:
            raise Exception('Bad scaler path')

        if 'name' not in data_dict or data_dict['name'] != 'km_scaler':
            raise Exception('Bad scaler dict')

        if 'std' not in data_dict or 'mean' not in data_dict:
            raise Exception('Bad scaler params')

        self.mean, self.std = data_dict['mean'], data_dict['std']


class Timing:
    def __init__(self, name):
        self.start_time = time.time()
        self.name = name

    def __enter__(self):
        print('%s...' % self.name)

    def __exit__(self, exc_type, exc_value, traceback):
        prc_time = time.time() - self.start_time
        print('Finished in %0.2F sec' % prc_time)


def binarize(x, thresh):
    x_bin = x.copy()
    x_bin[x_bin >= thresh] = 1.0
    x_bin[x_bin <  thresh] = 0.0
    return x_bin


def patch_borders(x, border_size, val):
    if border_size > 0:
        x[:border_size, :] = val
        x[:, :border_size] = val
        x[-border_size:, :] = val
        x[:, -border_size:] = val
    return x


def adf(x, y, border_size=0, val=0.0):
    return patch_borders(abs(x - y), border_size, val)
