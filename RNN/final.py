
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib as plt
import sklearn 
import numpy as np
import pandas as pd
import time

np.random.seed(9)
#計算程式執行時間
StartTime = time.time()

# Define the Preprocess function(input the raw csv files)
def Data_Preprocess(raw_data):
    #This function is main use to process the data for feed into the deep neural network

    #do the one hot encodeing,make the non-numeric features to numeric features
    data_one_hot = pd.get_dummies(data=raw_data, columns=["protocol_type"])
    data_one_hot = pd.get_dummies(data=data_one_hot, columns=["Service"])
    data_one_hot = pd.get_dummies(data=data_one_hot, columns=["flag"])

    #turn data list to data array
    data_array = data_one_hot.values

    #find the 'normal' label index that can do the replacement of this feature.   result label index:0
    '''i, = np.where(data_array[0]=='normal')'''

    #transfer the result to numeric type
    for x in range(len(data_array)):
        if data_array[x][0]!='normal.':
            data_array[x][0]=1
        else:
            data_array[x][0]=0

    #output the shape of the numpy data array
    '''print(data_array.shape)'''

    #output all of the label 
    Label = data_array[:,0]
    #output all of the feature
    Features = data_array[:,1:]
    
    #Normalization
    #import the module 
    '''
    from sklearn import preprocessing

    #use the preprocessing that can normalize the feature 
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

    #make the features transfer to 0~1
    scaled_Features = minmax_scale.fit_transform(Features)
    
    return scaled_Features, Label
    '''
    return Features, Label

#讀入檔案
#file_name = 'C:\\Users\Maxwu\Desktop\Tensorflow_works\Datasets\\NSL_KDD\KDDTrain+_Preprocess.xlsx'
#讀入檔案切割其中1000筆來測試
#file_name = '//Users/wudongye/Desktop/DeepLearningIDS/Datasets/KDDTrain+_Raw_1000.csv'#in OSX
file_name = 'C:\\Users\\MaxWu\\Documents\\GitHub\\DeepLearningIDS\\Datasets\\KDDcombined+_Raw.csv'#in Windows
#file_name = '//Users/wudongye/Documents/GitHub/DeepLearningIDS/Datasets/KDDTrain+_Raw_1000.csv' #in OSX
all_data = pd.read_csv(file_name)

all_Features, all_Label = Data_Preprocess(all_data)
mask = np.random.rand(len(all_data)) < 0.8
train_Features = all_Features[mask]
train_Label = all_Label[mask]
test_Features = all_Features[~mask]
test_Label = all_Label[~mask]


# In[2]:


print ("Train X shape is :", train_Features.shape)
print ("Train Y shape is :", train_Label.shape)
print ("Test X shape is :", test_Features.shape)
print ("Test Y shape is :", test_Label.shape)


# In[3]:


learning_rate = 0.001
n_classes = 1
display_step = 100
training_cycles = 1000
hidden_units = 3
input_features = 119
time_steps = 1


# In[4]:


newtrain_X = train_Features.reshape(len(train_Features), time_steps, input_features)
newtrain_Y = train_Label.reshape(len(train_Label), n_classes)
print(newtrain_X.shape)
print(newtrain_Y.shape)


# In[5]:


#Input Placeholders
x = tf.placeholder(tf.float32, [None, time_steps, input_features])
y = tf.placeholder(tf.float32, [None, n_classes])
print(x.shape)
print(y.shape)
#Weights and Biases
weights = {
    # (1, 3)
    'in': tf.Variable(tf.random_normal([time_steps, hidden_units])),
    # (3, 1)
    'out': tf.Variable(tf.random_normal([hidden_units, n_classes]))
}
biases = {
    # (3, )
    'in': tf.Variable(tf.constant(0.1, shape=[hidden_units, ])),
    # (1, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


# In[6]:


print(weights['in'].shape)
print(weights['out'].shape)
print(biases['in'].shape)
print(biases['out'].shape)


# In[7]:


def RNN(X, weights, biases):
    print(X.shape)
    X = tf.reshape(X, [-1, time_steps])
    print(X.shape)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    print(X_in.shape)
    X_in = tf.reshape(X_in, [-1, time_steps, hidden_units])
    print(X_in.shape)
    
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_units)
    init_state = cell.zero_state(len(newtrain_X), dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    print(outputs.shape)
    print(outputs[-1].shape)
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    #print(outputs.shape)
    print(outputs[-1].shape)
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    print(results.shape)
    return results


# In[8]:


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[9]:


correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[10]:


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


# In[11]:


for i in range (1000):
    sess.run([train_op], feed_dict = {x:newtrain_X, y:newtrain_Y})
    
    if (i) % 100 == 0:
        print ("Cost for the training cycle : ",i," : is : ",sess.run(cost, feed_dict ={x :newtrain_X,y:newtrain_Y}))
correct = tf.equal(tf.argmax(output, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
print('Accuracy on the overall test set is :',accuracy.eval({x:newtest_X, y:newtest_Y}))

