#將NSL-KDD取得的原始資料做必須的前處理

#NSL-KDD檔案名稱

file_name_osx = '/Users/wudongye/Documents/GitHub/DeepLearningIDS/Datasets/KDDTrain+_Raw_1000.csv'
file_name_win = 'C:\\Users\\Maxwu\\Documents\\GitHub\\DeepLearningIDS\\Datasets\\KDDTrain+_Raw_1000.csv'
#NSL_KDD_minus21_test  = 'C:\\Users\Maxwu\Documents\GitHub\DeepLearningIDS\Datasets\\NSL_KDD\KDDTest-21.txt'

#從NSL-KDD分類的資料中挑選所需的標籤(Labels)
wanted_feature = [0,1,2,5,6,23,24] #挑選的特徵
# 41: attack_or_not
# 0 : Duration
# 1 : Protocol_type
# 4 : Src_bytes
# 5 : dst_bytes
# 22: counts
# 23: srv_count

with open(file_name_win) as file_index:
    lines = file_index.readlines()
    
temp_data = []

for x in lines:
    sparse = x.split(',')
    temp_data.append([sparse[i] for i in wanted_feature])

#將挑選出來的標籤中的Protocol_types轉換為數字
for x in range(len(temp_data)):
	temp_data[x]=[w.replace('tcp','1').replace('udp','2').replace('icmp','3') for w in temp_data[x]]

#將挑選出來的標籤中的attack_or_not轉換為數字
for x in range(len(temp_data)):
    if temp_data[x][0]!='normal':
        temp_data[x][0]='1'
    else:
        temp_data[x][0]='0'

#輸出處理過後的檔案
output_file = 'Extracted_for_test_1000.csv'
temp_file = open(output_file,'w')
temp_file.write('LABEL,FEAT_1,FEAT_2,FEAT_3,FEAT_4,FEAT_5,FEAT_6')

#將所有處理過的資料寫入
for i in temp_data:
    	temp_file.write('\n' + ','.join(i))
temp_file.close()




