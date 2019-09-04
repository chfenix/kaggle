import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

labelEncoder = preprocessing.LabelEncoder()
oneHotEncoder = preprocessing.OneHotEncoder(sparse=False)

# 读取训练数据
data_train = pd.read_csv("./data/train.csv")
print("======Data Train Shape======\n", data_train.shape)

# 读取测试数据
data_test = pd.read_csv("./data/test.csv")
print("======Data Test Shape======\n", data_test.shape)

# 组合测试和训练集，组成全集
data_full = pd.concat([data_train, data_test], axis=0, sort=True)

# 查看SalePrice数据概览
print("======SalePrice Desc======\n", data_train["SalePrice"].describe())
print("======SalePrice Empty======\n" + str(data_train["SalePrice"].isnull().sum()))
# 查看SalePrice分布情况
# sns.distplot(data_train["SalePrice"])

# 数据清洗
# 查看整体数据情况
# 读取特征定义文件
f_pro_des = open("./data/pro_desc.csv", "r", encoding="gbk")
for line in f_pro_des.readlines():
    line = line.strip()
    pro_name = line.split(",")[0]   # 特征名称
    pro_type = line.split(",")[1]   # 特征类型

    # ========查看特征数据概况========
    if data_full.get(pro_name) is None:
        # 无效字段，跳过
        continue
    if pro_type == "":
        # 无类型信息，不做分析
        continue

    print("======", pro_name, "======")
    if pro_type == "E":
        # 枚举型，显示汇总统计
        print(pd.value_counts(data_full[pro_name]))
    if pro_type == "N":
        # 数值类型，显示统计信息
        print(data_full[pro_name].describe())
    print("Empty Value:", data_full[pro_name].isnull().sum())
    # ========查看特征数据概况========

    # ========空值填充========
    print("========= Data Cleaning =========")
    pro_null = line.split(",")[2]  # 空值处理方式(均值(Mean),中位数(Median),众数(Mode),emtpy(数值型填充0，枚举型填充none))
    if pro_null.lower() == "empty":
        if pro_type == "E":
            # 枚举型，填充None
            data_train[pro_name].fillna("None", inplace=True)
            data_test[pro_name].fillna("None", inplace=True)
        if pro_type == "N":
            # 数值类型，填充0
            data_train[pro_name].fillna(0, inplace=True)
            data_test[pro_name].fillna(0, inplace=True)

    # 均值填充
    if pro_null.lower() == "mean":
        data_train[pro_name].fillna(data_full[pro_name].mean(), inplace=True)
        data_test[pro_name].fillna(data_full[pro_name].mean(), inplace=True)

    # 中位数填充
    if pro_null.lower() == "median":
        data_train[pro_name].fillna(data_full[pro_name].median(), inplace=True)
        data_test[pro_name].fillna(data_full[pro_name].median(), inplace=True)

    if pro_null.lower() == "mode":
        data_train[pro_name].fillna(data_full[pro_name].mode().iloc[0], inplace=True)
        data_test[pro_name].fillna(data_full[pro_name].mode().iloc[0], inplace=True)
    # ========空值填充========

    # ========特征处理========
    print("========= Feature Process =========")
    pro_fe = line.split(",")[3]  # 特征处理方式（LB:LabelEncoder OH:OneHotEncoder）
    if pro_fe.lower() == "lb":
        # LabelEncoder
        data_train[pro_name] = labelEncoder.fit_transform(data_train[pro_name])
        data_test[pro_name] = labelEncoder.fit_transform(data_test[pro_name])

    if pro_fe.lower() == "oh":
        # OneHotEncoder
        data_train[pro_name] = oneHotEncoder.fit_transform(data_train[pro_name].values.reshape(-1, 1))
        data_test[pro_name] = oneHotEncoder.fit_transform(data_test[pro_name].values.reshape(-1, 1))
    # ========特征处理========

# 特征选择
print("========= Feature Selection =========")
# 计算训练集不同特性相关度 皮尔森相关系数
corr_mat = data_train.corr()
plt.subplots(figsize=(20, 9))
selection_num = 10  # 选取相关度最大的特征数目
valid_feature = corr_mat.nlargest(selection_num + 1, "SalePrice")["SalePrice"].index  # 根据SalePrice截取相关度最大的feNum个特征（不包含SalePrice）
cm = np.corrcoef(data_train[valid_feature].values.T)
sns.set(font_scale=1.25)
# 显示热力图
hm = sns.heatmap(cm, cbar=True, annot=True,
                 square=True, fmt=".2f", annot_kws={"size": 10}, yticklabels=valid_feature.values, xticklabels=valid_feature.values,
                 cmap="YlGnBu")
print(valid_feature)
plt.show()



