import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import csv
import time
import pandas as pd
import numpy as np

from keras.utils import np_utils        # keras np 的包
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

from keras import models
from keras import layers

file_addr = "../dataset/used_car_train_20200313/used_car_train_20200313.csv"

## 通过Pandas对于数据进行读取 (pandas是一个很友好的数据读取函数库)
Train_data = pd.read_csv('../dataset/used_car_train_20200313/used_car_train_20200313.csv', sep=' ')
TestA_data = pd.read_csv('../dataset/used_car_testB_20200421/used_car_testB_20200421.csv', sep=' ')


# Step 3:特征与标签构建
# 1) 提取数值类型特征列名
numerical_cols = Train_data.select_dtypes(exclude='object').columns
categorical_cols = Train_data.select_dtypes(include='object').columns

# 2) 构建训练和测试样本
## 选择特征列
feature_cols = [col for col in numerical_cols if col not in ['SaleID','name','regDate','creatDate','price','model','brand','regionCode','seller']]
feature_cols = [col for col in feature_cols if 'Type' not in col]

## 提前特征列，标签列构造训练样本和测试样本
X_data = Train_data[feature_cols]
Y_data = Train_data['price']

X_test = TestA_data[feature_cols]

X_data = X_data.fillna(-1)

X_data = X_data.values
Y_data = Y_data.values

mean = Y_data.mean(axis=0)
print(mean)

model = Sequential([
    Dense(units=32, input_dim=18, activation='sigmoid'),
    Dense(units=1, activation='sigmoid'),
])

# 定义优化器
sgd = SGD(lr=0.005)
adam = Adam(lr=0.005)

# 定义优化器， loss function， 训练过程中计算准确率
model.compile(
    optimizer=sgd,
    loss='mse',
    metrics=['accuracy']                                    # 加上 metrics 的 accuracy 后，输出会显示 accuracy 的值
)
# 训练模型
model.fit(X_data, Y_data, batch_size=100, epochs=10)                     # 每次放 32 张图片， 共 10 次迭代周期

# 评估模型
# loss, accuracy = model.evaluate(x_test, y_test)

# print('\ntest loss', loss)
# print('accuracy', accuracy)







