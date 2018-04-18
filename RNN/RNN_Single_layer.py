#import all the require libries
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib import rnn
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import matplotlib.pylab as plt
import time

start_time = time.time() #the time when program start

#Read the training file path
file_path = 'Processed_Data.csv'
train_data = pd.read_csv(file_path)

def parse_fl(data):
    # This function mainly to transfer the raw csv file to 
    # training features and labels. 
    
    # Select the features in the input data
    features = data[:, 122]
    # Select the labels in the input data
    labels = data[:, 0:122]
    #trans the label from text to

    return features, labels

# Prepare the training features and labels 
train_features, train_labels = trans_fl(train_data)
