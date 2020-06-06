# super-resolution-for-MRIs
Comparison of various super resolution techniques for MRIs
i
The dataset for training and testing was created from the following source:

https://brain-development.org/ixi-dataset/

This process involved taking a few middle slices out of the MRI and saving those slices as png.
Recreation of this dataset is possible by using the nii2png.py script.

The dataset function can be found in dataset.py and is used to load the low/high resolution
images into tensorflow as a dataset.

The models can be found in the files by their name
edsr -> edsr.py
wdsr -> wdsr.py
The structure is based on the papers linked below and the implementation is based off of the
work done by Martin Krasser. This implementation can be found here:
https://github.com/krasserm/super-resolution

The edsr model is from the following paper:
https://arxiv.org/abs/1707.02921

The wdsr model is from the following paper:
https://arxiv.org/abs/1808.08718

Weights from previous training instances can be found in the weights folder.
