import sklearn 
import numpy as np
import pandas as pd
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

    '''
    #transfer the result to numeric type
    for x in range(len(data_array)):
        if data_array[x][0]!='normal':
            data_array[x][0]=1
        else:
            data_array[x][0]=0
    '''

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
file_name = 'C:\\Users\Maxwu\Documents\GitHub\DeepLearningIDS\Datasets\KDDcombined+_Raw.csv'#in Windows
#file_name = '//Users/wudongye/Documents/GitHub/DeepLearningIDS/Datasets/KDDTrain+_Raw_1000.csv' #in OSX
all_data = pd.read_csv(file_name)


#distributed random normal slice the data into training and testing data
all_Features, all_Label = Data_Preprocess(all_data)
mask = np.random.rand(len(all_data)) < 0.8
train_Features = all_Features[mask]
train_Label = all_Label[mask]
test_Features = all_Features[~mask]
test_Label = all_Label[~mask]

print (test_Label.shape)

#print('volumn:',len(train_Features))
#print('dimention:',train_Features.shape)
train_Features = train_Features.reshape((-1, 1, 123))
test_Features = test_Features.reshape((-1, 1, 123))

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN

model = Sequential()
#model.add(Embedding(107, 107))
# model.add(Dense(units=32, input_dim= 123))
model.add(SimpleRNN(units=123, input_dim = 123))
model.add(Dense(5, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('\nTrain...\n')
model.fit(train_Features, train_Label,epochs=15,validation_data=(train_Features, train_Label))
score, acc = model.evaluate(test_Features, test_Label)
print('\nTest score:', score)
print('\nTest accuracy:', acc)

print('\n')
print('Run Time = %.2s seconds' % (time.time() - StartTime))
