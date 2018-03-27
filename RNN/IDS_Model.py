import tensorflow as tf
import numpy as np
import pandas as pd

# Read file 
train_file_path = 'C:\\Users\\MaxWu\\Documents\\GitHub\\DeepLearningIDS\\RNN\\Processed_Data_Train.csv'
train_data = pd.read_csv(file_path)

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

train_features, train_labels = trans_fl(all_data)
        

