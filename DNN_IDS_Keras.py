import sklearn 
import numpy as np
import pandas as pd
import time
np.random.seed(9)

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
    Feature = data_array[:,1:]

    #Normalization
    #import the module 
    from sklearn import preprocessing

    #use the preprocessing that can normalize the feature 
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

    #make the features transfer to 0~1
    scaled_Features = minmax_scale.fit_transform(Feature)

    return scaled_Features, Label

#讀入檔案
#file_name = 'C:\\Users\Maxwu\Desktop\Tensorflow_works\Datasets\\NSL_KDD\KDDTrain+_Preprocess.xlsx'
#讀入檔案切割其中1000筆來測試
file_name = '//Users/wudongye/Desktop/DeepLearningIDS/Datasets/KDDTrain+_Raw_1000.csv'
all_data = pd.read_csv(file_name)

#distributed random normal slice the data into training and testing data
all_Features, all_Label = Data_Preprocess(all_data)
mask = np.random.rand(len(all_data)) < 0.8
train_Features = all_Features[mask]
train_Label = all_Label[mask]
test_Features = all_Features[~mask]
test_Label = all_Label[~mask]
'''
print(train_Features.shape)
print(train_Label.shape)
print(test_Features.shape)
print(test_Label.shape)
'''

from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(units=40, input_dim=107, kernel_initializer='uniform', activation='relu'))

model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))

model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=train_Features, y=train_Label, 
                            validation_split=0.1, epochs=30, batch_size=30, 
                            verbose=2)

scores = model.evaluate(x=test_Features, y=test_Label)


#calculate the run time
'''
StartTime = time.time()
StopTime = time.time()
print('Run Time = %.2s' % (StopTime - StartTime))
'''

#print all the test data(from 1 to 1000)
'''
for i in range(1000):
    print('\n')
    print(all_data[i:i+1])
    time.sleep(1)
'''
