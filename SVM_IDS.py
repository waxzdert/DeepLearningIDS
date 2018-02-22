import numpy as np
import pylab as pl
from sklearn import svm


def feed_data(file_name):
    #從檔案中讀入所有資料(字串)，並取得資料總量(整數)
    with open(file_name) as file_index:
        lines = file_index.readlines()
    
    #從每筆資料當中切割出 'Label' 和 'Feature' 
    feature = []
    label = []
    x = []
    y = []
    for each_line in lines[1:]:
        label.append(each_line[0])
        feature.append(each_line[1:].strip().split(','))


    return (feature,label)



file_name1 = 'C:\\Users\\Maxwu\\Documents\\GitHub\\DeepLearningIDS\\Extracted_for_test.csv'

output_feature, ouput_label = feed_data(file_name1)
print(output_feature)
print(ouput_label)

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
