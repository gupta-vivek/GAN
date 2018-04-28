# -*- coding: utf-8 -*-
"""
    @created on: 3/4/18,
    @author: Vivek A Gupta,
    @version: v0.0.1

    Description:

    Sphinx Documentation Status:

    ..todo::
    
"""
import pandas as pd
import numpy as np


# One hot vector function.
def one_hot_vector(label, size):
    output_label = []
    for i in label:
        a = np.zeros(size, dtype='float')
        a[i] = 1
        output_label.append(a)
    return output_label


# Divide into batches.
def divide_batches(input_batch, batch_size):
    output_batch = []
    for i in range(0, len(input_batch), batch_size):
        output_batch.append(input_batch[i: i + batch_size])
    return output_batch


# Read data.
def read_data_csv():

    # Read training data from csv.
    df = pd.read_csv('data/mnist_train.csv')
    train_label = df.label
    df = df.drop(['label'], axis=1)
    train_data = df.values
    print("Size of training set - ", len(train_data))

    # Read testing data from csv.
    df = pd.read_csv('data/mnist_train.csv')
    test_label = df.label
    df = df.drop(['label'], axis = 1)
    test_data = df.values
    print("Size of testing set - ", len(test_data))

    # Convert labels to one hot vector.
    train_label_1 = one_hot_vector(train_label, 10)
    test_label_1 = one_hot_vector(test_label, 10)

    return train_data, train_label_1, test_data, test_label_1
