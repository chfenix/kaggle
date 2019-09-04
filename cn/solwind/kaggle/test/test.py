import pandas as pd
import numpy as np
from sklearn import preprocessing

# 读取训练数据
data_train = pd.read_csv("../housePrices/data/train.csv")
print("======Data Train Shape======\n", data_train.shape)

# 读取测试数据
data_test = pd.read_csv("../housePrices/data/test.csv")
print("======Data Test Shape======\n", data_test.shape)

# 组合测试和训练集，组成全集
data_full = pd.concat([data_train, data_test], axis=0, sort=True)

print(data_full.groupby(['MSSubClass'])[['SalePrice']].agg(['mean','median','count']))

list = []
for i in range(10):
    list.append("aa")
print(list)