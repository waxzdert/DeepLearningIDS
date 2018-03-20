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
        if data_array[x][0]!='normal':
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
file_name = 'C:\\Users\\MaxWu\\Documents\\GitHub\\DeepLearningIDS\\Datasets\\KDDTrain+_Raw_1000.csv'#in Windows
#file_name = '//Users/wudongye/Documents/GitHub/DeepLearningIDS/Datasets/KDDTrain+_Raw_1000.csv' #in OSX
all_data = pd.read_csv(file_name)


#distributed random normal slice the data into training and testing data
all_Features, all_Label = Data_Preprocess(all_data)
mask = np.random.rand(len(all_data)) < 0.8
train_Features = all_Features[mask]
train_Label = all_Label[mask]
test_Features = all_Features[~mask]
test_Label = all_Label[~mask]

#hyperparameter
learning_rate = 0.001
n_classes = 2
display_step = 100
input_features = train_Features.shape[1] #No of selected features(columns)
training_cycles = 1000
time_steps = 5 # No of time-steps to backpropogate
hidden_units = 50 #No of LSTM units in a LSTM Hidden Layer

#Input Placeholders
with tf.name_scope('input'):
    x = tf.placeholder(tf.float64,shape = [None,time_steps,input_features], name = "x-input")
    y = tf.placeholder(tf.float64, shape = [None,n_classes],name = "y-input")
#Weights and Biases
with tf.name_scope("weights"):
    W = tf.Variable(tf.random_normal([hidden_units,n_classes]),name = "layer-weights")

with tf.name_scope("biases"):
    b = tf.Variable(tf.random_normal([n_classes]),name = "unit-biases")

#Unstacking the inputs with time steps to provide the inputs in sequence
# Unstack to get a list of 'time_steps' tensors of shape (batch_size, input_features)
x_ = tf.unstack(x,time_steps,axis =1)

#Defining a single GRU cell
rnn_cell = rnn.BasicRNNCell(hidden_units)

#GRU Output
with tf.variable_scope('RNN'):
    rnnoutputs,rnnstates = rnn.static_rnn(rnn_cell,x_,dtype=tf.float64)
    
#Linear Activation , using gru inner loop last output
output =  tf.add(tf.matmul(rnnoutputs[-1],tf.cast(W,tf.float64)),tf.cast(b,tf.float64))

#Defining the loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits = output))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Training the Model
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range (training_cycles):
    _,c = sess.run([optimizer,cost], feed_dict = {x:train_Features, y:train_Label})
    
    if (i) % display_step == 0:
        print ("Cost for the training cycle : ",i," : is : ",sess.run(cost, feed_dict ={x :train_Features,y:train_Label}))
correct = tf.equal(tf.argmax(output, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
print('Accuracy on the overall test set is :',accuracy.eval({x:test_Features, y:test_Label}))

# Therefore, we are extracting the final labels => '1 0' = '1' = Normal (and vice versa)
# Steps to calculate the confusion matrix

pred_class = sess.run(tf.argmax(output,1),feed_dict = {x:test_Features,y:test_Label})
labels_class = sess.run(tf.argmax(y,1),feed_dict = {x:test_Features,y:test_Label})
conf = tf.contrib.metrics.confusion_matrix(labels_class,pred_class,dtype = tf.int32)
ConfM = sess.run(conf, feed_dict={x:test_Features,y:test_Label})
print ("confusion matrix \n",ConfM)

#Plotting the Confusion Matrix
labels = ['Normal', 'Attack']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(ConfM)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()