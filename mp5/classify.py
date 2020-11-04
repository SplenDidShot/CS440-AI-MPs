# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np


def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    weights = np.zeros_like(np.arange(len(train_set[0]), dtype=float))
    bias = 0.0
    for _ in range(max_iter):
        for i in range(len(train_set)):
            cur_image = np.array(train_set[i])
            activation_function = np.dot(weights, cur_image) + bias
            prediction = np.sign(activation_function)
            if prediction <= 0:
                prediction = 0
            else:
                prediction = 1

            if prediction != train_labels[i]:
                if train_labels[i] == 0:
                    y = -1
                else:
                    y = 1
                weights += learning_rate*y*cur_image
                bias += learning_rate*y

    return weights, bias


def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter=20):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    trained_weights, trained_bias = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    result_labels = []
    for i in range(len(dev_set)):
        cur_image = np.array(dev_set[i])
        activation_function = np.dot(trained_weights, cur_image) + trained_bias
        prediction = np.sign(activation_function)

        if prediction <= 0:
            prediction = 0
        else:
            prediction = 1

        result_labels.append(prediction)

    return result_labels


def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    result_labels = []
    from math import sqrt
    for i in range(len(dev_set)):
        cur_image = np.array(dev_set[i])
        distances = []
        for j in range(len(train_set)):
            train_image = np.array(train_set[j])
            dist = np.linalg.norm(cur_image-train_image)
            # if dist<20:
            distances.append((dist, train_labels[j]))
        distances.sort()
        denom = 0
        label_sum = 0
        for n in range(len(distances)):
            label_sum += distances[n][1]
            denom += 1
            if denom==k:
                break
        label_sum /= denom
        if label_sum > 0.5:
            result_labels.append(1)
        else:
            result_labels.append(0)

    return result_labels
