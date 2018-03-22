import tensorflow as tf
import pandas as pd
import math

file = 'C:\\Users\\MaxWu\\Documents\\GitHub\\DeepLearningIDS\\Datasets\\KDDTrain+.csv'
raw_data = pd.read_csv(file)

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
    #
    # new Xi = ((old Xi)-min)/(Max-min)
    pass

OneHotData = one_hot(raw_data)
fir_norm(OneHotData)

OneHotData.to_csv('Processed_Data.csv')