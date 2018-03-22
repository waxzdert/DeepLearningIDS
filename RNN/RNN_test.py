
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


learning_rate = 0.001
n_classes = 1
display_step = 100
training_cycles = 1000
hidden_units = 3
input_features = 119
time_steps = 1


# In[21]:


newtrain_X = train_Features.reshape(len(train_Features), time_steps, input_features)
newtrain_Y = train_Label.reshape(len(train_Label), n_classes)

# In[5]:


#Input Placeholders
with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32,shape = [None,time_steps,input_features], name = "x-input")
    y = tf.placeholder(dtype=tf.float32, shape = [None,n_classes],name = "y-input")
#Weights and Biases
with tf.name_scope("weights"):
    W = tf.Variable(tf.random_normal([hidden_units,n_classes]),name = "layer-weights")

with tf.name_scope("biases"):
    b = tf.Variable(tf.random_normal([n_classes]),name = "unit-biases")


# In[7]:


rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_units)


# In[32]:


with tf.variable_scope('RNN'):
    rnnoutputs,rnnstates = tf.nn.dynamic_rnn(rnn_cell,x,dtype=tf.float32)


# In[33]:


output = tf.add(tf.matmul(rnnoutputs[-1],tf.cast(W,tf.float32)),tf.cast(b,tf.float32))


# In[34]:


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=output))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[27]:


sess = tf.InteractiveSession()


# In[28]:


sess.run(tf.global_variables_initializer())
sess.run(output)


# In[25]:


for i in range (training_cycles):
    _,c = sess.run([optimizer,cost], feed_dict = {x:newtrain_X, y:newtrain_Y})
    
    if (i) % display_step == 0:
        print ("Cost for the training cycle : ",i," : is : ",sess.run(cost, feed_dict ={x :newtrain_X,y:newtrain_Y}))
correct = tf.equal(tf.argmax(output, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

