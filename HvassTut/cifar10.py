########################################################################
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call load_class_names() to get an array of the class-names.
# 3) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 4) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import pickle
import os
#import download
from dataset import one_hot_encoded
#import tensorflow as tf 
import sys

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
#Numberoftrainingsamples = 15
#Numberoftestsamples = 5250
# Number of images for each batch-file in the training-set.
#_images_per_file = int(Numberoftrainingsamples/3)
#data_path = "D:/Iheya_n/HvassTutResults/1BJaBPlSCr/3to1ratio/{}Train/num_iter_128000_{}Test/".format(Numberoftrainingsamples, Numberoftestsamples) #"data/CIFAR-10/"
#data_path = "D:/Iheya_n/HvassTutResults/2BatSCrNSe/3to1ratio/{}Train/num_iter_10000_{}Test/".format(Numberoftrainingsamples, Numberoftestsamples)
#data_path = "D:/Iheya_n/HvassTutResults/2BatSCrNSe/NewResults/3to1ratio/{}Train/".format(Numberoftrainingsamples)
# URL for the data-set on the internet.
#data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 50 #32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 3 #10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 3 #5

## Total number of images in the training-set.
## This is used to pre-allocate arrays for efficiency.
#_num_images_train = _num_files_train * _images_per_file

########################################################################
# Private functions for downloading, unpacking and loading data-files.

label_bytes = 1  # 2 for CIFAR-100
record_bytes = label_bytes + img_size_flat

def _get_file_path(filename,data_path1):
    """
    Return the full path of a data-file for the data-set.

    If filename=="" then return the directory of the files.
    """

    return os.path.join(data_path1, filename)


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename,data_path1,data_path):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """
    file_path = _get_file_path(filename,data_path1)
    print("Loading data: " + file_path)
    data = open(file_path, 'r')
    string = data.read()
    stringtolist = [int(s) for s in string[:-1].split(",")]
    openfile=open(data_path + "results.txt","a")
    openfile.write("Loading data: " + file_path + "\n")
    # Get the raw images.
#    raw_images = data[b'data']
    copyofstringtolist = list(stringtolist)
    del copyofstringtolist[::record_bytes]
    raw_images = copyofstringtolist

    # Get the class-numbers for each image. Convert to numpy-array.
#    cls = np.array(data[b'labels'])
    clslist = (stringtolist[::record_bytes])
    cls = np.array(clslist)

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls
#to check
#len(images)=5250, images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
#                           np.zeros(shape=[15750, 50, 50, 3], dtype=float)
#cls = np.zeros(shape=[_num_images_train], dtype=int)

def load_class_names(data_path1):
    """
    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """
   
    with open(data_path1 + "batches.meta.txt","r") as f:
#        lines = txt.reader(f, delimiter="\n")
#        lines = f.readlines()
#        lines = [line[:-1] for line in f]
        lines = [line.rstrip('\n') for line in f]
    return lines
#    print (lines)
#    return lines #why this doesnt work? return func


def load_training_data(data_path1,_num_images_train,data_path):
    """
    The data-set is split into 3 data-files which are merged here.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1) + ".txt",data_path1=data_path1,data_path=data_path) ###".bin"

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data(data_path1,data_path):
    """
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = _load_data(filename="test_batch.txt",data_path1=data_path1,data_path=data_path) ###.bin

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

########################################################################
