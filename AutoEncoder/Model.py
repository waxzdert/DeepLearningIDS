import tensorflow as tf
import numpy as np
import pandas as pd

# Input Data
train_file_path = '/Users/wudongye/Documents/GitHub/DeepLearningIDS/RNN/Processed_Data_Train.csv'
test_file_path = '/Users/wudongye/Documents/GitHub/DeepLearningIDS/RNN/Processed_Data_Test.csv'

# Prepare the 
def parse_fl(data):
    # This function mainly to transfer the raw csv file to 
    # training features and labels. 
    
    # Select the features in the input data
    features = data[:, 0:122]
    # Select the labels in the input data
    labels = data[:, 122]
    #trans the label from text to

    return features, labels
# Read file
train_data = pd.read_file(train_file_path)
test_data = pd.read_file(test_file_path)

# Tranfer the file form PandasDataframe to numpy ndarray
train_data = train_data.values
test_data = test_data.values

# Prepare the training features and labels 
train_Features, train_Labels = parse_fl(train_data)
test_Features, test_Labels = parse_fl(test_data)

# Parameters
learning_rate = 0.01
training_epoch = 5
display_step = 1
example_to _show = 10

# Network parameter
N_input = 122
