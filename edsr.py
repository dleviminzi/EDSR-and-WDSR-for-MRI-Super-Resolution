from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, BatchNormalization
from tensorflow.python.keras.models import Model

from common import normalize, denormalize, pixel_shuffle


def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b_in  = b
        b = Conv2D(num_filters, 3, padding='same', activation='relu')(b)
        b = Conv2D(num_filters, 3, padding='same')(b)
        if res_block_scaling:
            b = Lambda(lambda t: t * res_block_scaling)(b)
        b = Add()([b_in, b])

    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")

# currently there is only a scale 4 model
def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    x = upsample_1(x, 2, name='conv2d_1_scale_2')
    x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x
