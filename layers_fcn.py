import numpy as np
import numpy.random as nr
import tensorflow as tf

multiply = tf.keras.layers.multiply
add = tf.keras.layers.add
Lambda = tf.keras.layers.Lambda
Conv2D = tf.keras.layers.Conv2D
ActivityRegularization = tf.keras.layers.ActivityRegularization
Layer = tf.keras.layers.Layer


def normal_noise(w, ratio=0.01):
    return nr.normal(0.0, ratio * (w.std() + 1), w.shape).astype(w.dtype)


def apply_noise(model, ratio=0.01):
    for l_conf in model.get_config()['layers']:
        layer = model.get_layer(l_conf['name'])
        weights = layer.get_weights()
        weights = [w + normal_noise(w, ratio) for w in weights]
        layer.set_weights(weights)
    return model


def SquareLayer(name=None):
    return Lambda(lambda x: tf.square(x), name=name)


def MeanFilterLayer(kernel_size, name=None):
    return Conv2D(
        filters=1,
        kernel_size=kernel_size,
        use_bias=False,
        padding='same',
        name=name,
        trainable=False,
        kernel_initializer='ones',
        bias_initializer='zeros'
    )


def sobel_filter():
    grad_v = np.expand_dims([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], axis=2)
    grad_h = np.expand_dims([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], axis=2)
    return np.concatenate((grad_v, grad_h), axis=2).tolist()


def GradientLayer(trainable=False, name=None):
    return Conv2D(
        filters=2,
        kernel_size=(3, 3),
        use_bias=False,
        padding='same',
        name=name,
        trainable=trainable,
        kernel_initializer=tf.keras.initializers.Constant(sobel_filter()),
        bias_initializer='zeros'
    )


def gauss_filter(ksize):
    sigma = (ksize - 1) / 6

    x, y = np.meshgrid(range(ksize), range(ksize))

    x0, y0 = (ksize - 1) / 2, (ksize - 1) / 2
    d = (x - x0) ** 2 + (y - y0) ** 2
    g = np.exp(-(d / (2.0 * sigma ** 2)))

    g /= g.sum()
    g = np.expand_dims(g, 3)
    g = np.expand_dims(g, 4)
    g = np.expand_dims(g, 0)
    return g.tolist()


def GaussFilterLayer(kernel_size, trainable=False, name=None):
    return Conv2D(
        filters=1,
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer=tf.keras.initializers.Constant(
            gauss_filter(kernel_size)),
        use_bias=False,
        padding='same',
        name=name,
        trainable=trainable
    )


def Conv1x1(kernel_initializer, use_bias=False,
            bias_initializer=0.0, activation=None,
            name=None, trainable=False):
    return Conv2D(
        filters=1,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=tf.keras.initializers.Constant(kernel_initializer),
        use_bias=use_bias,
        bias_initializer=tf.keras.initializers.Constant(bias_initializer),
        activation=activation,
        name=name,
        trainable=trainable
    )


def get_slice(idx, name=None):
    return Lambda(
        lambda x: tf.expand_dims(x[:, :, :, idx], axis=-1),
        name=name)


def dir_filters():
    ver = np.asarray([[0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=bool)
    hor = np.asarray([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=bool)
    dg1 = np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=bool)
    dg2 = np.asarray([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=bool)

    t1 = np.tan(np.pi/8)
    t3 = np.tan(3 * np.pi/8)

    dir_dict = [
        dict(name='0',    min_angle=-t1, max_angle=t1,  filter=hor),  # 0
        dict(name='pi4',  min_angle=t1,  max_angle=t3,  filter=dg1),  # pi/4
        dict(name='pi2',  min_angle=t3,  max_angle=-t3, filter=ver),  # pi/2
        dict(name='3pi4', min_angle=-t3, max_angle=-t1, filter=dg2)   # 3pi/4
    ]

    return dir_dict


class NonMaximumSuppression(Layer):
    def __init__(self, mask=None, **kwargs):
        self.mask = mask

        self.dfilt = np.expand_dims(self.mask, axis=3)

        zero = tf.zeros_like(self.dfilt, dtype=tf.float32)
        minf = tf.constant(
            value=-np.inf,
            dtype=tf.float32,
            shape=self.dfilt.shape,
            name='minus_inf'
        )

        # perform warning-free log of boolean mask: 0.0 -> -inf; 1.0 -> 0.0
        # self.mask = np.log(mask)
        self.dfilt = tf.where(self.dfilt, zero, minf)
        super(NonMaximumSuppression, self).__init__(**kwargs)

    def build(self, input_shape):
        super(NonMaximumSuppression, self).build(input_shape)

    def call(self, x, **kwargs):
        masked_max = tf.nn.dilation2d(x, self.dfilt, padding='SAME',
                                      strides=[1, 1, 1, 1],
                                      rates=[1, 1, 1, 1])

        gr_mask = tf.greater(x, masked_max)
        suppressed = tf.cast(gr_mask, x.dtype) * x
        return suppressed

    def get_config(self):
        config = super(NonMaximumSuppression, self).get_config()
        config.update(dict(mask=self.mask.tolist()))
        return config


class Bias(Layer):
    def __init__(self, activation=None, bias_initializer='zeros',
                 bias_regularizer=None, bias_constraint=None, **kwargs):
        self.bias = None
        self.activation = tf.keras.activations.get(activation)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(input_shape[-1],),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        super(Bias, self).build(input_shape)

    def call(self, x, **kwargs):
        outputs = tf.nn.bias_add(x, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


def relu(x):
    return tf.where(x >= 0.0, x, tf.zeros_like(x))


def relu_t(x, t):
    mint = t * tf.ones_like(x, name='minus_threshold')
    return tf.where(x >= 0, x, mint)


class ThresholdLayer(Layer):
    def __init__(self,  activation=None, bias_initializer='zeros',
                 bias_regularizer=None, bias_constraint=None, **kwargs):
        self.bias = None
        self.activation = tf.keras.activations.get(activation)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        super(ThresholdLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(input_shape[-1],),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        super(ThresholdLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        outputs = relu_t(tf.nn.bias_add(x, self.bias), self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


def clipped_div(x, y, clip_val):
    eps = tf.ones_like(y, name='eps') * 1e-9
    y_tr = tf.where(tf.equal(y, 0.0), eps, y)
    div = tf.divide(x, y_tr)
    div_clipped = tf.clip_by_value(div, -clip_val, clip_val)
    return div_clipped


class GradientDirection(tf.keras.layers.Layer):
    def __init__(self, activation=None, **kwargs):
        self.activation = tf.keras.activations.get(activation)
        super(GradientDirection, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GradientDirection, self).build(input_shape)

    def call(self, x, **kwargs):
        div = clipped_div(x[:, :, :, 0], x[:, :, :, 1], 10.0)
        div = tf.expand_dims(div, axis=3)
        if self.activation is not None:
            return self.activation(div)
        return div

    def compute_output_shape(self, input_shape):
        return input_shape[:-1], 1


class ClipValues(tf.keras.layers.Layer):
    def __init__(self, min_value=None, max_value=None, **kwargs):
        self.min_value = min_value
        self.max_value = max_value
        super(ClipValues, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ClipValues, self).build(input_shape)

    def call(self, x, **kwargs):
        if self.min_value < self.max_value:
            ret = (self.min_value <= x) & (x < self.max_value)
        else:
            ret = (self.min_value <= x) | (x < self.max_value)
        return tf.cast(ret, x.dtype)

    def get_config(self):
        config = super(ClipValues, self).get_config()
        config.update(dict(min_value=self.min_value, max_value=self.max_value))
        return config
