import numpy as np
import tensorflow as tf
import sys
from PIL import Image as PImage

def exitProgram(errStr, i):
	sys.stderr.write(errStr+'\n')
	sys.exit(i)

def load_image(filename):
    with open(filename) as f:
        return np.array(f.read())
    
def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)

def main():
    if len(sys.argv) != 1:
        exitProgram("Error: wrong number of argument.", 1)
    
    # img1 = PImage.open(sys.argv[1])
    # img2 = PImage.open(sys.argv[2])

    # im1 = load_image('./raisr_img1_before.png')
    # im2 = load_image('./raisr_img1_after.png')
    im1 = load_image('./output_source/323016.png')
    im2 = load_image('./output_source/323016_result.png')

    img1 = tf.image.decode_png(im1)
    img2 = tf.image.decode_png(im2)

    # img1 = tf.image.convert_image_dtype(im1, tf.float32)
    # img2 = tf.image.convert_image_dtype(im2, tf.float32)
    print(psnr(img1, img2))

    sys.exit(0)

if __name__ == '__main__':
    main()
