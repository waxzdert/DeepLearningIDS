from keras.layers import Input, Dense
from keras.models import Model
import pandas as pd

#Read the training file path
file_dict = {
    'Train':'Processed_Data_Train.csv',
    'Test':'Processed_Data_Test.csv',
    'Minus21':'Processed_Data_Minus21.csv'
}
train_data = pd.read_csv(file_dict['Train'])
test_data = pd.read_csv(file_dict['Test'])
minus_data = pd.read_csv(file_dict['Minus21'])

def parse_fl(data):
    # This function mainly to transfer the raw csv file to 
    # training features and labels. 
    
    # Select the features in the input data
    features = data[:, 0:122]
    # Select the labels in the input data
    labels = data[:, 122]
    #trans the label from text to

    return features, labels

train_data = train_data.values
test_data = test_data.values
minus_data = minus_data.values

# Prepare the training features and labels 
train_Features, train_Labels = parse_fl(train_data)
test_Features, test_Labels = parse_fl(test_data)
minus_Features, minus_Labels = parse_fl(minus_data)

input_data = Input(shape=(122,))
encoded = Dense(60, activation='relu')(input_data)
encoded = Dense(30, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
temp = encoded
decoded = Dense(30, activation='relu')(encoded)
decoded = Dense(60, activation='relu')(decoded)
decoded = Dense(122, activation='sigmoid')(decoded)

autoencoder = Model(input_data, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(train_Features,train_Features,
                    epochs=10,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(test_Features, test_Features))

