import cv2
import os

lr_dir = './lr/'
files = os.listdir(lr_dir)

num_files = len(files)

intensity = 0

for file in files:
    img = cv2.imread(lr_dir + file)
    W = img.shape[0]
    H = img.shape[1]
    n = W * H
    for xi in range(H):
        for yi in range(W):
            intensity += img[xi, yi]

denom = n * num_files

intensity = intensity/denom

print(intensity*255)

