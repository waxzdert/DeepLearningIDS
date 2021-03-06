import pandas as pd
import math

file_dict = {
    'Train':'C:\\Users\\MaxWu\\Documents\\GitHub\\DeepLearningIDS\\Datasets\\KDDTrain+.csv',
    'Test':'C:\\Users\\MaxWu\\Documents\\GitHub\\DeepLearningIDS\\Datasets\\KDDTest+.csv',
    'Minus21':'C:\\Users\\MaxWu\\Documents\\GitHub\\DeepLearningIDS\\Datasets\\KDDTest-21.csv'
}

#計算資料的特徵中有多少獨立項
#print(raw_data.groupby('protocol_type').ngroups)
#print(raw_data.groupby('Service').ngroups)
#print(raw_data.groupby('flag').ngroups)


#================== OneHot Encoding ==================
def one_hot(in_df):

    # Convert some nonnumeric features into numeric form
    # Example: protocol_type, service and flag

    #將prorocol type 轉換為數值, dim: 41->44, (125973, 44)
    oh_df = pd.get_dummies(data=in_df, columns=['protocol_type'])
    #將Service轉換為數值, dim: 44->112, (125973, 112)
    oh_df = pd.get_dummies(data=oh_df, columns=['Service'])
    #將flag轉換為數值, dim: 112->122,(125973, 122)
    oh_df = pd.get_dummies(data=oh_df, columns=['flag'])
    return oh_df

#================== Normalization ==================
def fir_norm(in_df):
    # Normalize the data which difference between maximum and minimum values
    # such as : ‘duration[0,58329]’,‘src_bytes[0,1.3 × 109]’ and ‘dst_bytes
    # Logarithmic scaling method for scaling to obtain the ranges
    # (x,y) -> (log(x),log(y))
    for i in range(len(in_df['Duration'])):
        if (in_df.loc[i,('Duration')]) == 0:
            in_df.loc[i,('Duration')] = 0
        else:
            in_df.loc[i,('Duration')] = round(math.log(in_df.loc[i,('Duration')],10), 2)

    for i in range(len(in_df['src_bytes'])):
        if (in_df.loc[i,('src_bytes')]) == 0:
            in_df.loc[i,('src_bytes')] = 0
        else:
            in_df.loc[i,('src_bytes')] = round(math.log(in_df.loc[i,('src_bytes')],10), 2)

    for i in range(len(in_df['dst_bytes'])):
        if (in_df.loc[i,('dst_bytes')]) == 0:
            in_df.loc[i,('dst_bytes')] = 0
        else:
            in_df.loc[i,('dst_bytes')] = round(math.log(in_df.loc[i,('dst_bytes')],10), 2)

def sec_norm(in_df):
    # Normalize all the data in the dataframe
    # let the data in the frame can 
    # new Xi = ((old Xi)-min)/(Max-min)

    from sklearn import preprocessing
    # remove the feature with string which can't normalize
    temp_data = in_df.drop('result', axis=1)

    # let data transform the type from dataframe to numpy array
    temp_data = temp_data.values

    # Define a scaler that will feed nraw data afterward
    # The scaler will normalize the data into new Xi = ((old Xi)-min)/(Max-min)
    # And it's output will range from 0 to 1.
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

    # feed the data to the scaler
    temp_data = scaler.fit_transform(temp_data)

    # Transfer the numpy array to dataframe
    temp_data = pd.DataFrame(temp_data)

    # merge the feature which remove first
    temp_data = temp_data.join(in_df[['result']])

    return temp_data

def ren_idx(new_df, old_df):
    # This function main use to rename the index of the dataset
    temp = []
    for i in range(1,len(old_df.columns)):
        temp.append(old_df.columns[i])
    temp.append(old_df.columns[0])
    new_df.columns = temp

def label_trans(in_df):
    # 1 represent the traffic is an attack
    # 0 represent the traffic is a normal traffic
    for i in range(len(in_df['result'])):
        if(in_df.loc[i,('result')]) != 'normal':
            in_df.loc[i,('result')] = 1
        else:
            in_df.loc[i,('result')] = 0

def execute(raw_data, idx):
    OneHotData = one_hot(raw_data)
    fir_norm(OneHotData)
    Processed_Data = sec_norm(OneHotData)
    ren_idx(Processed_Data, OneHotData)
    label_trans(Processed_Data)
    Processed_Data.to_csv('Processed_Data_%s.csv' % (idx),index=False,index_label=False)

train_data = pd.read_csv(file_dict['Train'])
execute(train_data, 'Train')

test_data = pd.read_csv(file_dict['Test'])
execute(test_data, 'Test')

Minus21 = pd.read_csv(file_dict['Minus21'])
execute(Minus21, 'Minus21')