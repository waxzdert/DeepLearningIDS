import numpy as np
#import pylab as pl
from sklearn import svm
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
        #print(feature[i])
        #print(label[i])
        #time.sleep(1)

    return (feature, label)

start_time = time.time()

file_name_win = 'C:\\Users\\Maxwu\\Documents\\GitHub\\DeepLearningIDS\\Extracted_for_test.csv'
file_name_osx = '/Users/wudongye/Documents/GitHub/DeepLearningIDS/Extracted_for_test.csv'

output_feature, ouput_label = feed_data(file_name_win)
#print(output_feature)
#print(ouput_label)

clf = svm.SVC(kernel = 'linear')
clf.fit(output_feature, ouput_label)

print('\n')
print(clf)
print('\n')
print(clf.support_vectors_)
print('\n')
print(clf.support_)
print('\n')
print(clf.n_support_)

print('\n')
print('-------------------------------------------')
print('\n')
runtime = time.time() - start_time
print('Run time ：', runtime,' sec')