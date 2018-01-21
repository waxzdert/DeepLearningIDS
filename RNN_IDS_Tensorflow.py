import tensorflow as tf
from tensorflow.contrib import rnn
import sklearn 
import numpy as np
import pandas as pd
import time

np.random.seed(9)
#計算程式執行時間
StartTime = time.time()

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
#file_name = 'C:\\Users\Maxwu\Documents\GitHub\DeepLearningIDS\Datasets\KDDcombined+_Raw.csv'#in Windows
file_name = '//Users/wudongye/Documents/GitHub/DeepLearningIDS/Datasets/KDDTrain+_Raw_1000.csv' #in OSX
all_data = pd.read_csv(file_name)


#distributed random normal slice the data into training and testing data
all_Features, all_Label = Data_Preprocess(all_data)
mask = np.random.rand(len(all_data)) < 0.8
train_Features = all_Features[mask]
train_Label = all_Label[mask]
test_Features = all_Features[~mask]
test_Label = all_Label[~mask]


#Training parameter
learning_rate = 0.001
training_steps = 10000
batch_size = 100
display_step = 20

# Network Parameters
num_input = 107 # NSL-KDD Dataset data input (img shape: 1*107)
num_hidden = 128 # hidden layer num of features
num_classes = 1 # NSL-KDD Dataset total classes

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x, weights, biases):
    
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    #x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(len(train_Features)):
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: train_Features[step], Y:train_Label})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: test_Features,
                                                                 Y: test_Label})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

