import csv
import time
import pandas as pd

file_addr = "../dataset/used_car_train_20200313/used_car_train_20200313.csv"

input_array = []
pred_array = []

# with open(file_addr, "r") as f:
#     reader = csv.reader(f)
#
#     for data in reader:
#         print(data)
#
#         for

## 通过Pandas对于数据进行读取 (pandas是一个很友好的数据读取函数库)
Train_data = pd.read_csv("../dataset/used_car_train_20200313/used_car_train_20200313.csv", sep=' ')
# TestA_data = pd.read_csv('datalab/231784/used_car_testA_20200313.csv', sep=' ')

## 输出数据的大小信息
print('Train data shape:', Train_data.shape)
print(Train_data.describe())
# print('TestA data shape:', TestA_data.shape)