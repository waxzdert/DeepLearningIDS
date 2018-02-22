import pandas as pd

file_name = '//Users/wudongye/Documents/GitHub/DeepLearningIDS/Datasets/KDDTrain+_Raw_1000.csv' #in OSX
all_data = pd.read_csv(file_name)

#data_one_hot = pd.get_dummies(data=all_data, columns=["result"])

data_array = all_data.values

print(data_array.shape)