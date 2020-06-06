import os
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE


def dataset(batch_size=16, repeat_count=None, random_transform=True, subset='train'):

    lr_dir = './dataset/lr/'
    hr_dir = './dataset/hr/'

    if subset == 'train':
        img_range = range(1, 901)
    elif subset == 'valid':
        img_range = range(901, 1021)
    else:
        raise ValueError('subset must be train or validate')

    hr_img_files = [os.path.join(hr_dir, f'{img_id:04}.png') for img_id in img_range]
    lr_img_files = [os.path.join(lr_dir, f'{img_id:04}.png') for img_id in img_range]

    hr_ds = tf.data.Dataset.from_tensor_slices(hr_img_files)
    hr_ds = hr_ds.map(tf.io.read_file)
    hr_ds = hr_ds.map(lambda x: tf.image.decode_png(x), num_parallel_calls=AUTOTUNE)

    lr_ds = tf.data.Dataset.from_tensor_slices(lr_img_files)
    lr_ds = lr_ds.map(tf.io.read_file)
    lr_ds = lr_ds.map(lambda x: tf.image.decode_png(x), num_parallel_calls=AUTOTUNE)

    ds = tf.data.Dataset.zip((lr_ds, hr_ds))


#    if random_transform:
#        ds = ds.map(lambda lr, hr: random_crop(lr, hr, scale=4), num_parallel_calls=AUTOTUNE)
#        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
#        ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds



##############################################################################
'''
krasserm transforms
'''
##############################################################################


def random_crop(lr_img, hr_img, hr_crop_size=96, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


































