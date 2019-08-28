# -*- coding:utf8 -*-
from collections import Counter

import matplotlib
import pandas
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns

from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

#  分析数据

# 读取训练集
data_train = pd.read_csv("../../../resources/train.csv")

# 销售价分布
print(data_train.SalePrice.describe())
# sns.distplot(data_train['SalePrice'])
# plt.title("SalePrice Distribution")

# 读取属性定义,生成属性类型字典
dict_pro_type = {}
f_pro_desc = open("../../../resources/pro_desc.csv", "r", encoding="gbk")
for line in f_pro_desc.readlines():
    line = line.strip()
    k = line.split(',')[0]
    v = line.split(',')[1]
    dict_pro_type[k] = v

f_pro_desc.close()


# 查询属性和售价的分布关系
_proName = 'GrLivArea'     # 属性名称
_proType = dict_pro_type[_proName]  # 属性类型 E:枚举 N:数值

# 组合属性和售价数据，进行绘图查看
data = pd.concat([data_train['SalePrice'], data_train[_proName]], axis=1)

if _proType == "E":
    # 枚举类型统计
    print(Counter(data_train[_proName]))
    fig = sns.boxplot(x=_proName, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)

if _proType == "N":
    # 数值类型统计
    print(data_train[_proName].describe())
    data.plot.scatter(x=_proName, y='SalePrice', ylim=(0, 800000))

# 对枚举类型进行数值化
f_names = ['CentralAir', 'Neighborhood']
for x in f_names:
    label = preprocessing.LabelEncoder()
    data_train[x] = label.fit_transform(data_train[x])

# 绘制关系矩阵
corrmat = data_train.corr()
f, ax = plt.subplots(figsize=(20, 9))
# sns.heatmap(corrmat, vmax=0.8, square=True)

k  = 10 # 关系矩阵中将显示10个特征
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index  # 根据SalePrice截取相关度最大的k行
cm = np.corrcoef(data_train[cols].values.T)
sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True,
#                   square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,cmap="YlGnBu")

# 绘制选中列关系矩阵点图
sns.set()
cols = ['SalePrice','OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
# sns.pairplot(data_train[cols], height = 2.5, kind="reg")

# 获取数据
# cols = ['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
cols = ['GrLivArea']
x = data_train[cols].values
print(x)
y = data_train['SalePrice'].values
x_scaled = preprocessing.StandardScaler().fit_transform(x)
print(x_scaled)
y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))


# plt.show()
