import pandas as pd
import math

file_dict = {
    'Train':'C:\\Users\\MaxWu\\Documents\\GitHub\\DeepLearningIDS\\AutoEncoder\\Processed_Data_Train.csv',
    'Test':'C:\\Users\\MaxWu\\Documents\\GitHub\\DeepLearningIDS\\AutoEncoder\\Processed_Data_Test.csv',
    'Minus21':'C:\\Users\\MaxWu\\Documents\\GitHub\\DeepLearningIDS\\AutoEncoder\\Processed_Data_Minus21.csv'
}

train_data = pd.read_csv(file_dict['Train'])

def parse(in_df, a_type):
    temp_list = []
    for i in range(len(in_df['result'])):
        if(in_df.loc[i,'result']) == a_type:
            temp_list.append(train_data.loc[i])
    
    tf = pd.DataFrame(temp_list)
    tf.to_csv("training_data_%s.csv" % a_type)

parse(train_data, 'dos')
parse(train_data, 'u2r')
parse(train_data, 'r2l')
parse(train_data, 'probe')
