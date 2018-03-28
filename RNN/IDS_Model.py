import tensorflow as tf
import numpy as np
import pandas as pd

# Read file 
train_file_path = 'C:\\Users\\MaxWu\\Documents\\GitHub\\DeepLearningIDS\\RNN\\Processed_Data_Train.csv'
train_data = pd.read_csv(train_file_path)

#test_file_path = ''
#test_data = 


def trans_fl(data):
    # This function mainly to transfer the raw csv file to 
    # training features and labels. 
    
    # Select the features in the input data
    features = data[:, 122]
    # Select the labels in the input data
    labels = data[:, 0:122]

    return features, labels

# Prepare the training features and labels 
train_features, train_labels = trans_fl(train_data)

# Hyperparameters
lr = 0.001 # Learning Rate : 優化函式降低loss的速度
classes = 1 # the model can classify 0 is normal, 1 is attack
hidden_units = 3 # The hidden units in the recurrent neural network
input_shape = 119 # The shape feeded in the neural network
time_steps = 1 # The steps of the input vector sequence

# Prepare the placeholder that will feed in the neural network 
X = tf.placeholder(dtype=tf.float64, shape=[None, time_steps, input_shape], name='X')
Y = tf.placeholder(dtype=tf.float64, shape=[None, classes, name='Y'])

# Define the Weight of the neuron i the neural network
W = tf.Variable(tf.random_normal())
