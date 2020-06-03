import os
import nibabel as nib
import cv2
# this script will require a seperate environment with deprecated scipy
# from scipy.misc import imresize only run with scipy==1.2.0


def slicer(img_type):
    '''
    takes middle slices of nii formated scans from rawdata directory and writes
    them into png format images

    types -> t1, t2, dti, pd, mra

    working:    t2 (w/o IOP labelled), dti, mra
    broken:     t1, pd
    '''

    files = os.listdir('./rawdata/{}nii/'.format(img_type))

    for file in files:
        if file[7] == 'I':
            continue
        try:
            img = nib.load('./rawdata/{}nii/'.format(img_type)+file)
            img = img.get_fdata()
            s = img.shape
            upper_bound = s[2]//2 + 10
            lower_bound = upper_bound - 20
            for i in range(lower_bound, upper_bound, 10):
                img_slice = img[:, :, i]
                cv2.imwrite('./dataset/{}/{}{}{}.png'.format(img_type, img_type[1], file[3:7], i), img_slice)
        except:
            print("{} does not have correct dimensions".format(file))


def lrhr(scale=4, lr_dir='./dataset/lr/', hr_dir='./dataset/hr/'):
    downgrades image by downscale and upscale
    i = 0
    for file in os.listdir(hr_dir):
        img = cv2.imread(hr_dir + file)
        H, W, cm = img.shape
        img_l = imresize(img, 1/scale, interp='bicubic')
        #img_l = imresize(img_l, (H, W), interp='bicubic')
        cv2.imwrite(lr_dir + '{}.png'.format(format(i, '04d'), scale), img_l)
        cv2.imwrite(hr_dir + '{}.png'.format(format(i, '04d')), img)
        i += 1


