import tensorflow as tf
import numpy as np
import time 

#初始時間
start_time = time.time()

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def Weight(shape):
    #初始化權重
    Weight = tf.random_normal(shape, mean=0.0, stddev=0.1)
    
    return tf.Variable(Weight)

def ForwardPropagation(X, W_1, W_2, W_3, W_4):
    hidden1 = tf.nn.sigmoid(tf.matmul(X, W_1))
    hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, W_2))
    hidden3 = tf.nn.sigmoid(tf.matmul(hidden2, W_3))
    y_hat = tf.matmul(hidden3, W_4)
    
    return y_hat

def Load_data(file_name):
    with open(file_name) as file_index:
        lines = file_index.readlines()
    keys = lines[0].strip().split(',')
    data = []
    for l in lines[1:]:
        data.append(dict(zip(keys, l.strip().split(','))))
    
    return data

def One_hot_encode(y_value):
    new_list=[]
    for val in y_value:
        if val==[0.0]:
            new_list.append([1,0])
        else:
            new_list.append([0,1])
    
    return 

def feed_data(label_dict):
    x = []
    y = []
    x_keys = ['FEAT_1','FEAT_2','FEAT_3','FEAT_4','FEAT_5','FEAT_6']
    y_keys = ['LABEL'];
    for item in label_dict:
        x.append([float(item[k]) for k in x_keys])
        y.append([float(item[k]) for k in y_keys])
   # y=One_Hot(y)
    
    return (x,y)

#設定訓練資料
training_file_name = '/Users/wudongye/Documents/GitHub/DeepLearningIDS/Extracted_for_train.txt'
training_data_dict = Load_data(training_file_name)

#設定測試資料
testing_file_name  = '/Users/wudongye/Documents/GitHub/DeepLearningIDS/Extracted_for_train.txt'

testing_data_dict  = Load_data(testing_file_name)

train_X, train_y = feed_data(training_data_dict)
test_X, test_y = feed_data(testing_data_dict)

#設定Ｘ,y,和隱藏層的神經元數量
x_size = (len(train_X[0]))
hidden1_nodes = 8
hidden2_nodes = 2
hidden3_nodes = 8
y_size = (len(train_y[0]))

#預留等等要傳入的資料的形狀
X = tf.placeholder("float", shape=[None, x_size])
y = tf.placeholder("float", shape=[None, y_size])

#初始化權重
W_1 = Weight((x_size, hidden1_nodes))
W_2 = Weight((hidden1_nodes, hidden2_nodes))
W_3 = Weight((hidden2_nodes, hidden3_nodes))
W_4 = Weight((hidden3_nodes, y_size))

#傳值進入神經網路
y_hat = ForwardPropagation(X, W_1, W_2, W_3, W_4)

#比較輸出和標籤(Label)的誤差
constant_value = tf.constant(0.5)
predict = tf.cast(tf.less(constant_value, y_hat), tf.float32)

#倒傳遞更新權重
cost = tf.squared_difference(y_hat, y)
updates = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

#執行TesnorFlow計算圖(Computational graph)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

result_file = 'result.txt'
for epoch in range(20):
    for i in range(len(train_X)):
        sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
    train_accuracy = np.mean(train_y == sess.run(predict, feed_dict={X:train_X, y: train_y}))
    test_accuracy  = np.mean(test_y == sess.run(predict, feed_dict={X:test_X, y: test_y}))
    print("Epoch = %d, train accuracy = %.2f, test accuracy = %.2f" % (epoch + 1, 100. * train_accuracy, 100. *test_accuracy))

    temp_file = open(result_file,'w')
    temp_file.write("Epoch = %d, train accuracy = %.2f, test accuracy = %.2f\n" % (epoch + 1, 100. * train_accuracy, 100. *test_accuracy))

print("--- %s seconds ---" % (time.time() - start_time))
temp_file.write("RunTime: %s seconds" % (time.time() - start_time))
temp_file.close()
