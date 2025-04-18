# -*- coding: utf-8 -*-
"""
@author: serdarhelli
"""

import os
import numpy as np
from PIL import Image
from zipfile import ZipFile
from natsort import natsorted


def convert_one_channel(img):
    # Some images have 3 channels, although they are grayscale
    if len(img.shape) > 2:
        return img[:, :, 0]
    else:
        return img

def pre_images(resize_shape, path, include_zip):
    # Handle Pillow version compatibility
    if hasattr(Image, 'Resampling'):
        resample_mode = Image.Resampling.LANCZOS
    else:
        resample_mode = Image.ANTIALIAS

    if include_zip:
        ZipFile(os.path.join(path, "DentalPanoramicXrays.zip")).extractall(path)
        path = os.path.join(path, 'Images')

    dirs = natsorted(os.listdir(path))
    sizes = np.zeros((len(dirs), 2))

    # Load first image
    img = Image.open(os.path.join(path, dirs[0]))
    sizes[0, :] = img.size
    img_resized = img.resize(resize_shape, resample=resample_mode)
    images = convert_one_channel(np.asarray(img_resized))

    for i in range(1, len(dirs)):
        img = Image.open(os.path.join(path, dirs[i]))
        sizes[i, :] = img.size
        img_resized = img.resize(resize_shape, resample=resample_mode)
        img_array = convert_one_channel(np.asarray(img_resized))
        images = np.concatenate((images, img_array))

    images = np.reshape(images, (len(dirs), resize_shape[0], resize_shape[1], 1))
    return images, sizes
