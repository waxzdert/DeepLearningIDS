import pandas as pd
import numpy as np 
import time

def feed_data(file_name):
    #從檔案中讀入所有資料(字串)，並取得資料總量(整數)
    with open(file_name) as file_index:
        lines = file_index.readlines()
    
    #從每筆資料當中切割出 'Label' 和 'Feature' 
    temp = []
    feature = []
    label = []
    for each_line in lines[1:]:
        label.append(each_line[0])
        feature.append(each_line[2:].strip().split(','))

    '''
    temp.append(eval(feature[0][0]))
    temp.append(eval(feature[0][1]))
    temp.append(eval(feature[0][2]))
    temp.append(eval(feature[0][3]))
    temp.append(eval(feature[0][4]))
    temp.append(eval(feature[0][5]))

    print(tmep)
    '''
    print(feature[0])
    return (feature, label)


file_name1 = 'C:\\Users\\Maxwu\\Documents\\GitHub\\DeepLearningIDS\\Extracted_for_test.csv'

output_feature, ouput_label = feed_data(file_name1)
#print(output_feature)
#print(ouput_label)






