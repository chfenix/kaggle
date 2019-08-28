import pandas as pd
import numpy as np

# 读取训练数据
data_train = pd.read_csv("./data/train.csv")
print("======Data Train Shape======\n", data_train.shape)

# 查看SalePrice数据概览
print("======SalePrice Desc======\n", data_train["SalePrice"].describe())
print("======SalePrice Empty======\n" + str(data_train["SalePrice"].isnull().sum()))
# 查看SalePrice分布情况
# sns.distplot(data_train["SalePrice"])

# 数据清洗
# 查看数据情况
# 读取特征定义文件
f_pro_des = open("./data/pro_desc.csv", "r", encoding="gbk")
for line in f_pro_des.readlines():
    # 轮询特征名称，查看特征数据概况
    line = line.strip()
    pro_name = line.split(',')[0]   # 特征名称
    pro_type = line.split(',')[1]   # 特征类型

    if data_train.get(pro_name) is None:
        # 无效字段，跳过
        continue
    if pro_type == "":
        # 无类型信息，不做分析
        continue

    print("======", pro_name, "======")
    if pro_type == "E":
        # 枚举型，显示汇总统计
        print(pd.value_counts(data_train[pro_name]))
    if pro_type == "N":
        # 数值类型，显示统计信息
        print(data_train[pro_name].describe())
    print("Empty Value:", data_train[pro_name].isnull().sum())



# plt.show()
