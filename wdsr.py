import tensorflow_addons as tfa

from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model

from common import normalize, denormalize, pixel_shuffle


'''
krasserm based implementation of the wdsr model
'''

def wdsr(scale, num_filters=32, num_res_blocks=8, res_block_expansion=6, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)

    # residual block
    linear = 0.8
    kernel_size = 3
    for i in range(num_res_blocks):
        m_in = m
        m = conv2d_weightnorm(num_filters*res_block_expansion, 1, padding='same', activation='relu')(m_in)
        m = conv2d_weightnorm(int(num_filters*linear), 1, padding='same')(m)
        m = conv2d_weightnorm(num_filters, kernel_size, padding='same')(m)
        if res_block_scaling:
            m = Lambda(lambda t: t * res_block_scaling)(m)
        m = Add()([m_in, m])

    m = conv2d_weightnorm(3 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    m = Lambda(pixel_shuffle(scale))(m)

    s = conv2d_weightnorm(3 * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    x = Lambda(denormalize)(x)

    return Model(x_in, x, name="wdsr")


def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)

