import sys
from subprocess import Popen
from multiprocessing import cpu_count
from signal import sigwait, SIGINT
from datetime import datetime

import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib

from utils import *


class Model:
    def __init__(self, root_path, model=None, wait_tb=True):
        self.root_path = root_path
        self.model = model
        self.session = self.get_tf_session()
        self.device = self.get_tf_device_name()
        self.scaler = Scaler()
        self.tb_process = None
        self.wait_tb = wait_tb

    @staticmethod
    def get_tf_session(graph=None):
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        # config.log_device_placement = True

        # config.inter_op_parallelism_threads = 1
        # config.intra_op_parallelism_threads = 1

        session = tf.Session(config=config, graph=graph)
        tf.keras.backend.set_session(session)
        return session

    @staticmethod
    def get_tf_device_name():
        local_devices = device_lib.list_local_devices()
        gpu_list = [x.name for x in local_devices if x.device_type == 'GPU']
        cpu_list = [x.name for x in local_devices if x.device_type == 'CPU']

        if len(gpu_list) > 0:
            return gpu_list[0]
        elif len(cpu_list) > 0:
            return cpu_list[0]
        return None

    @staticmethod
    def run_tensorboard(model_path):
        logs_path = os.path.join(model_path, 'tb_logs')

        tb_path = os.path.join(sys.exec_prefix, 'bin', 'tensorboard')
        tb_process = Popen([tb_path, '--logdir', logs_path, '--port', '6007'])
        return tb_process

    def model_path(self, model_name):
        return os.path.join(self.root_path, model_name)

    def set_model(self, new_model):
        self.model = new_model

    @staticmethod
    def get_default_callbacks(model_path):
        # reduce_cb = callbacks.ReduceLROnPlateau(factor=0.8, patience=50)
        nan_cb = tf.keras.callbacks.TerminateOnNaN()

        weights_path = os.path.join(model_path, 'weights')
        make_dir(weights_path)
        save_path = os.path.join(weights_path, 'weights.{epoch:03d}.h5')
        save_cb = tf.keras.callbacks.ModelCheckpoint(save_path)

        log_path = os.path.join(model_path, 'log.csv')
        log_cb = tf.keras.callbacks.CSVLogger(log_path)

        tb_log_path = os.path.join(model_path, 'tb_logs')
        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=tb_log_path)

        return [tb_cb, nan_cb, save_cb, log_cb]

    def summary(self):
        return self.model.summary()

    def compile(self, optimizer, loss, metrics=None):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def __fit(self, data, epochs, batch_size, initial_epoch, callbacks=None):
        if callbacks is None:
            callbacks = []

        time_suffix = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        print(time_suffix)
        model_path = self.model_path(time_suffix)
        make_dir(model_path)

        callbacks += self.get_default_callbacks(model_path)
        x_train, y_train, x_val, y_val = data

        # x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0,
        # validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
        # steps_per_epoch=None, validation_steps=None

        if self.tb_process:
            self.tb_process.terminate()
        self.tb_process = self.run_tensorboard(model_path)

        history = self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            initial_epoch=initial_epoch,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=2
        )

        return history

    def __fit_generator(self, generator, val_generator=None, epochs=1,
                        initial_epoch=0, callbacks=None, verbose=1):
        if callbacks is None:
            callbacks = []

        time_suffix = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        print(time_suffix)
        model_path = self.model_path(time_suffix)
        make_dir(model_path)

        callbacks += self.get_default_callbacks(model_path)

        if self.tb_process:
            self.tb_process.terminate()
        self.tb_process = self.run_tensorboard(model_path)

        history = self.model.fit_generator(
            generator=generator,
            epochs=epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=None,  # len(Sequence)
            validation_data=val_generator,
            validation_steps=None,  # len(Sequence)
            callbacks=callbacks,
            verbose=verbose,
            class_weight=None,  # default
            max_queue_size=10,  # default
            use_multiprocessing=True,
            workers=cpu_count(),
            shuffle=True
        )

        return history

    def __run_in_session(self, fn, **kwargs):
        with self.session.as_default(), tf.device(self.device):
            result = fn(**kwargs)
        return result

    def fit(self, data, epochs=400, batch_size=32, initial_epoch=0, callbacks=None):
        return self.__run_in_session(
            self.__fit,
            data=data,
            epochs=epochs,
            batch_size=batch_size,
            initial_epoch=initial_epoch,
            callbacks=callbacks
        )

    def fit_generator(self, generator, val_generator=None, epochs=400,
                      initial_epoch=0, callbacks=None, verbose=1):
        return self.__run_in_session(
            self.__fit_generator,
            generator=generator,
            val_generator=val_generator,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=verbose
        )

    def __evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=0)

    def evaluate(self, x, y):
        return self.__run_in_session(self.__evaluate, x=x, y=y)

    def __normalize_and_predict(self, x):
        x = x.astype(np.float32) / 255
        return self.__predict(x)

    def __predict(self, x):
        n_dims = len(x.shape)
        if n_dims < 4:
            x = np.expand_dims(x, 0)
        if n_dims < 3:
            x = np.expand_dims(x, -1)
        # return np.squeeze(self.model.predict(x))
        return self.model.predict(x)

    def predict(self, x):
        return self.__run_in_session(self.__predict, x=x)

    def normalize_and_predict(self, x):
        return self.__run_in_session(self.__normalize_and_predict, x=x)

    def save(self, path):
        self.model.save(path)

    def load(self, path, custom_objects=None):
        self.model = tf.keras.models.load_model(path, compile=True, custom_objects=custom_objects)

    def load_by_best(self, name='val_loss', path=None, custom_objects=None):
        if not path:
            all_models = sorted(os.listdir(self.root_path))
            if not all_models:
                raise Exception('No models found')
            path = os.path.join(self.root_path, all_models[-1])

        log_path = os.path.join(path, 'log.csv')

        log = pd.read_csv(log_path, index_col=0)
        best_epoch = log[name].idxmin()
        print('loading epoch #%d' % best_epoch)
        weights_name = 'weights.%03d.h5' % (best_epoch + 1)

        weights_path = os.path.join(path, 'weights', weights_name)
        self.load(weights_path, custom_objects)

    def __del__(self):
        if not self.tb_process or self.tb_process.poll():
            return

        if self.wait_tb:
            print('Finished, press Ctrl+C to exit TensorBoard', file=sys.stderr)
            sigwait([SIGINT])

        self.tb_process.terminate()
