from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import sys

import random

#Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

import cifar10

#from cifar10 import img_size, num_channels, num_classes
# Width and height of each image.
img_size = 50 #32
# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3
# Number of classes.
num_classes = 3 #10

img_size_cropped = 24 #

def plot_images(images, cls_true, cls_pred=None, smooth=True, label_pred=None):
    if len(images)==len(cls_true)==16:
        fig,axes = plt.subplots(4,4)
    else:
        assert len(images)==len(cls_true)

        if len(images)==0:
            pass
        elif len(images)/3-int(len(images)/3)==0:
            fig,axes = plt.subplots(3,int(len(images)/3))
        elif len(images)/2-int(len(images)/2)==0:
            fig,axes = plt.subplots(2,int(len(images)/2))
        else:
            fig,axes = plt.subplots(1,int(len(images)))
    if len(images)==0:
        pass
    else:
        if cls_pred is None:
            hspace = 0.3
        else:
            hspace = 1.0
        fig.subplots_adjust(hspace=hspace, wspace=1.5)

    if len(images)==0:
        pass
    elif len(images)==1:
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'
        ax = axes
        ax.imshow(images[0, :, :, :],
                   interpolation=interpolation)
        # Name of the true class.
        cls_true_name = class_names[cls_true[0]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[0]]
            np.set_printoptions(precision=3, suppress=True)
            xlabel = "True: {0}\nPred: {1}\n{2}".format(cls_true_name, cls_pred_name, label_pred[0])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        #plt.show()
    else:
        for i, ax in enumerate(axes.flat):
            # Interpolation type.
            if smooth:
                interpolation = 'spline16'
            else:
                interpolation = 'nearest'

            # Plot image.
            ax.imshow(images[i, :, :, :],
                      interpolation=interpolation)
            
            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]
                np.set_printoptions(precision=3, suppress=True)
                xlabel = "True: {0}\nPred: {1}\n{2}".format(cls_true_name, cls_pred_name, label_pred[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
        
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
            # Ensure the plot is shown correctly with multiple plots
            # in a single Notebook cell.
            #plt.show()
    
    mm = 0
    while os.path.exists(data_path + 'figure_{}.png'.format(mm)):
        mm += 1
    plt.savefig(data_path + 'figure_{}.png'.format(mm))

# Get the first images from the test-set.
#images = images_test[0:9]

# Get the true classes for those images.
#cls_true = cls_test[0:9]

# Plot the images and labels using our helper-function above.
#plot_images(images=images, cls_true=cls_true, smooth=False)
#plot_images(images=images, cls_true=cls_true, smooth=True)

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    
    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image

def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images

distorted_images = pre_process(images=x, training=True)

def main_network(images, training):
    # Wrap the input images as a Pretty Tensor object.
    x_pretty = pt.wrap(images)

    # Pretty Tensor uses special numbers to distinguish between
    # the training and testing phases.
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    # Create the convolutional neural network using Pretty Tensor.
    # It is very similar to the previous tutorials, except
    # the use of so-called batch-normalization in the first layer.
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=num_classes, labels=y_true)

    return y_pred, loss

def create_network(training):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    with tf.variable_scope('network', reuse=not training):
        # Just rename the input placeholder variable for convenience.
        images = x

        # Create TensorFlow graph for pre-processing.
        images = pre_process(images=images, training=training)

        # Create TensorFlow graph for the main processing.
        y_pred, loss = main_network(images=images, training=training)

    return y_pred, loss

_, loss = create_network(training=True)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

y_pred, _ = create_network(training=False)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.

    with tf.variable_scope("network/" + layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable

weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

def get_layer_output(layer_name):
    # The name of the last operation of the convolutional layer.
    # This assumes you are using Relu as the activation-function.
    tensor_name = "network/" + layer_name + "/Relu:0"

    # Get the tensor with this name.
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

    return tensor

output_conv1 = get_layer_output(layer_name='layer_conv1')
output_conv2 = get_layer_output(layer_name='layer_conv2')

session = tf.Session()

def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False) #Generate a uniform random sample from np.arrange(num_images) of size (train_batch_size) without replacement

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch

# Best validation accuracy seen so far.
best_validation_accuracy = 0.0

# Iteration-number for last improvement to validation accuracy.
last_improvement = 0

# Stop optimization if no improvement found in this many iterations.
#require_improvement = 1500

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations,require_improvement):
    # Ensure we update the global variables rather than local copies.
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        
        # Increase the total number of iterations performed.
        # It is easier to update it in each iteration because
        # we need this number several times in the following.
        total_iterations += 1

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations and after last iteration.
        if (total_iterations % 100 == 0) or (i == (num_iterations - 1)):

            # Calculate the accuracy on the training-batch.
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)

            # Calculate the accuracy on the validation-set.
            # The function returns 2 values but we only need the first.
            acc_validation, _ = validation_accuracy()

            # If validation accuracy is an improvement over best-known.
            if acc_validation > best_validation_accuracy:
                # Update the best-known validation accuracy.
                best_validation_accuracy = acc_validation
                
                # Set the iteration for the last improvement to current.
                last_improvement = total_iterations

                # Save all variables of the TensorFlow graph to file.
                saver.save(sess=session, save_path=save_path)

                # A string to be printed below, shows improvement found.
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''
            
            # Status-message for printing.
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"

            # Print it.
            print(msg.format(i + 1, acc_train, acc_validation, improved_str))
            openfile.write(msg.format(i + 1, acc_train, acc_validation, improved_str) + "\n")

        # If no improvement found in the required number of iterations.
        if total_iterations - last_improvement > require_improvement:
            msg1 = ("No improvement found in a while, stopping optimization at {}".format(total_iterations))
            print (msg1)
            openfile.write(msg1 + "\n")
            # Break out from the for-loop.
            break

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    msg2 = ("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    print (msg2)
    openfile.write(msg2 + "\n")

def plot_example_errors(cls_pred, correct, label_pred):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    # Get the images from the test-set that have been incorrectly classified.
    images1 = images_test[incorrect]
    # Get the predicted classes for those images.
    cls_pred1 = cls_pred[incorrect]
    # Get the true classes for those images.
    cls_true1 = cls_test[incorrect]
    label_pred_1 = label_pred[incorrect]

    images2 = images_test[correct]
    cls_pred2 = cls_pred[correct]
    cls_true2 = cls_test[correct]
    label_pred_2 = label_pred[correct]

    t0p0 = []
    t0p1 = []
    t0p2 = []
    t1p0 = []
    t1p1 = []
    t1p2 = []
    t2p0 = []
    t2p1 = []
    t2p2 = []

    n = 0
    for i in cls_true2:
        if i == 0:
            t0p0.append(n)
        elif i == 1:
            t1p1.append(n)
        elif i == 2:
            t2p2.append(n)
        n = n+1

    m = 0
    for i in cls_true1:
        if i == 0:
            if cls_pred1[m] == 1:
                t0p1.append(m)
            else:
                t0p2.append(m)
        elif i == 1:
            if cls_pred1[m] == 0:
                t1p0.append(m)
            else:
                t1p2.append(m)
        elif i == 2:
            if cls_pred1[m] == 0:
                t2p0.append(m)
            else:
                t2p1.append(m)
        m = m+1

    truevspredicted1 = [t0p1, t0p2, t1p0, t1p2, t2p0, t2p1]
    for k in truevspredicted1:
        if len(k)>=16:
            randomsamples = [int(i) for i in random.sample(k,16)]
        else:
            randomsamples = [int(i) for i in random.sample(k,len(k))]
        plot_images(images=np.asarray([images1[i] for i in randomsamples]),
                    cls_true=np.asarray([cls_true1[i] for i in randomsamples]),
                    cls_pred=np.asarray([cls_pred1[i] for i in randomsamples]),
                    smooth=True,
                    label_pred=np.asarray([label_pred_1[i] for i in randomsamples]))
    truevspredicted2 = [t0p0, t1p1, t2p2]
    for k in truevspredicted2:
        if len(k)>=16:
            randomsamples = [int(i) for i in random.sample(k,16)]
        else:
            randomsamples = [int(i) for i in random.sample(k,len(k))]
        plot_images(images=np.asarray([images2[i] for i in randomsamples]),
                    cls_true=np.asarray([cls_true2[i] for i in randomsamples]),
                    cls_pred=np.asarray([cls_pred2[i] for i in randomsamples]),
                    smooth=True,
                    label_pred=np.asarray([label_pred_2[i] for i in randomsamples]))

    #get 3 pics of 25 pics of each true 0,1,2 species
#    randomsamples = [int(i) for i in random.sample(range(len(images1)),16)]
    
    #targetedsamples = [int(i) for i in [0,1,2,int(len(images))/2,int(len(images))/2+1,int(len(images)/2)+2,int(len(images))-1,int(len(images))-2,int(len(images))-3]]
    # Plot the first 9 images.

#    plot_images(images=np.asarray([images1[i] for i in randomsamples]), #images[0:9],
#                cls_true=np.asarray([cls_true1[i] for i in randomsamples]), #cls_true[0:9],
#                cls_pred=np.asarray([cls_pred1[i] for i in randomsamples])) #cls_pred[0:9])


def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)
        print(cm[i, :], class_name, file=openfile)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))
    openfile.write("".join(class_numbers)+"\n")


# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256

def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    label_pred = np.zeros(shape=(num_images,3))
    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}
        #feed_dict_2 = {x: images[i:j, :]}
        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        label_pred[i:j] = session.run(y_pred, feed_dict={x: images[i:j, :]})
        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred, label_pred

def predict_cls_test():
    return predict_cls(images = images_test,
                       labels = labels_test,
                       cls_true = cls_test)

def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.
    
    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()

def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _, _L = predict_cls_test()
    
    # Calculate the classification accuracy and return it.
    return classification_accuracy(correct)

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred, label_pred = predict_cls_test()
    
    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)
    
    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))
    openfile.write(msg.format(acc, num_correct, num_images)+"\n")
    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        openfile.write("Example errors:\n")
        plot_example_errors(cls_pred=cls_pred, correct=correct, label_pred=label_pred)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        openfile.write("Confusion Matrix:\n")
        plot_confusion_matrix(cls_pred=cls_pred)

def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print statistics for the weights.
    print("Min:  {0:.5f}, Max:   {1:.5f}".format(w.min(), w.max()))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
    openfile.write("Min:  {0:.5f}, Max:   {1:.5f}\n".format(w.min(), w.max()))
    openfile.write("Mean: {0:.5f}, Stdev: {1:.5f}\n".format(w.mean(), w.std()))
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)
    abs_max = max(abs(w_min), abs(w_max))

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=-abs_max, vmax=abs_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    #plt.show()
    mm = 0
    while os.path.exists(data_path + 'figure_{}.png'.format(mm)):
        mm += 1
    plt.savefig(data_path + 'figure_{}.png'.format(mm))

def plot_layer_output(layer_output, image):
    # Assume layer_output is a 4-dim tensor
    # e.g. output_conv1 or output_conv2.

    # Create a feed-dict which holds the single input image.
    # Note that TensorFlow needs a list of images,
    # so we just create a list with this one image.
    feed_dict = {x: [image]}
    
    # Retrieve the output of the layer after inputting this image.
    values = session.run(layer_output, feed_dict=feed_dict)

    # Get the lowest and highest values.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    values_min = np.min(values)
    values_max = np.max(values)

    # Number of image channels output by the conv. layer.
    num_images = values.shape[3]

    # Number of grid-cells to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_images))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid image-channels.
        if i<num_images:
            # Get the images for the i'th output channel.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, vmin=values_min, vmax=values_max,
                      interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    #plt.show()
    mm = 0
    while os.path.exists(data_path + 'figure_{}.png'.format(mm)):
        mm += 1
    plt.savefig(data_path + 'figure_{}.png'.format(mm))

def plot_distorted_image(image, cls_true):
    # Repeat the input image 9 times.
    image_duplicates = np.repeat(image[np.newaxis, :, :, :], 9, axis=0)

    # Create a feed-dict for TensorFlow.
    feed_dict = {x: image_duplicates}

    # Calculate only the pre-processing of the TensorFlow graph
    # which distorts the images in the feed-dict.
    result = session.run(distorted_images, feed_dict=feed_dict)

    # Plot the images.
    plot_images(images=result, cls_true=np.repeat(cls_true, 9))

def get_test_image(i):
    return images_test[i, :, :, :], cls_test[i]

def plot_image(image):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 2)

    # References to the sub-plots.
    ax0 = axes.flat[0]
    ax1 = axes.flat[1]

    # Show raw and smoothened images in sub-plots.
    ax0.imshow(image, interpolation='nearest')
    ax1.imshow(image, interpolation='spline16')

    # Set labels.
    ax0.set_xlabel('Raw')
    ax1.set_xlabel('Smooth')
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    #plt.show()
    mm = 0
    while os.path.exists(data_path + 'figure_{}.png'.format(mm)):
        mm += 1
    plt.savefig(data_path + 'figure_{}.png'.format(mm))

for i in [15,42,81,156,501,1002,4500,6750,9000,11250,13500,15750]:
    for j in ["TrialValidation1","TrialValidation2","TrialValidation3"]:
        #xxx = "Don't run"
        xxx = "Run"
        num_iterations = 1000000
        train_batch_size = 64
        imgchosen = 2473 #should be SCr
        Numberoftrainingsamples = i
        Numberoftestsamples = 5250
        require_improvement = 1500
        _images_per_file = int(Numberoftrainingsamples/3)
        _num_files_train = 3 #5
        _num_images_train = _num_files_train * _images_per_file
        data_path = "D:/Iheya_n/HvassTutResults/2BatSCrNSe/NewResults/3to1ratio/{}Train/{}/".format(Numberoftrainingsamples,j) #data_path = cifar10.data_path
        data_path1 = "D:/Iheya_n/HvassTutResults/2BatSCrNSe/NewResults/3to1ratio/{}Train/".format(Numberoftrainingsamples)
        class_names = cifar10.load_class_names(data_path1)
        images_train, cls_train, labels_train = cifar10.load_training_data(data_path1,_num_images_train,data_path)
        images_test, cls_test, labels_test = cifar10.load_test_data(data_path1,data_path)

        openfile = open(data_path + "results.txt", "a")

        print("Size of:")
        print("- Training-set:\t\t{}".format(len(images_train)))
        print("- Test-set:\t\t{}".format(len(images_test)))
        openfile.write("size of:\n- Training-set:\t\t{}\n- Test-set:\t\t{}".format(len(images_train),len(images_test)))

        #save_dir = data_path + 'checkpoints/'
        save_dir = data_path + 'checkpoints/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, 'cifar10_cnn')

        try:
            print("Trying to restore last checkpoint ...")
            openfile.write("Trying to restore last checkpoint ...\n")
            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

            # Try and load the data in the checkpoint.
            saver.restore(session, save_path=last_chk_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
            openfile.write("Restored checkpoint from:", last_chk_path)
        except:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint. Initializing variables instead.")
            openfile.write("Failed to restore checkpoint. Initializing variables instead.")
            session.run(tf.global_variables_initializer())
    
       img, cls = get_test_image(imgchosen)
        plot_distorted_image(img, cls)

        print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=True)
        plot_conv_weights(weights=weights_conv1, input_channel=0)
        plot_conv_weights(weights=weights_conv2, input_channel=1)

        if xxx == "Don't run":
            if False:
                optimize(num_iterations=10000,require_improvement)
        else:
            optimize(num_iterations,require_improvement)

        print_test_accuracy(show_example_errors=True,
                            show_confusion_matrix=True)
        plot_conv_weights(weights=weights_conv1, input_channel=0)
        plot_conv_weights(weights=weights_conv2, input_channel=1)

        plot_image(img)
        plot_layer_output(output_conv1, image=img)
        plot_layer_output(output_conv2, image=img)

        label_pred, cls_pred = session.run([y_pred, y_pred_cls],
                                           feed_dict={x: [img]})

        # Set the rounding options for numpy.
        np.set_printoptions(precision=3, suppress=True)

        # Print the predicted label.
        print(label_pred[0])
        print(label_pred[0],file=openfile)

plt.close()
        # This has been commented out in case you want to modify and experiment
        # with the Notebook without having to restart it.
session.close()
openfile.close()