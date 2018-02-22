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
    
    for i in range(len(feature)):
        for j in range(len(feature[i])):
            temp.append(eval(feature[i][j]))
        #print(temp)
        feature[i] = temp
        temp = []
        print(feature[i])
        print(label[i])
        time.sleep(1)


    return (feature, label)


file_name_win = 'C:\\Users\\Maxwu\\Documents\\GitHub\\DeepLearningIDS\\Extracted_for_test.csv'
file_name_osx = '/Users/wudongye/Documents/GitHub/DeepLearningIDS/Extracted_for_test.csv'

output_feature, ouput_label = feed_data(file_name_osx)
#print(output_feature)
#print(ouput_label)






