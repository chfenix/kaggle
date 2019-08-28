import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取训练数据
data_train = pd.read_csv("./data/train.csv")
# print(data_train)

# 查看SalePrice数据概览
print("SalePrice Desc\n" + str(data_train["SalePrice"].describe()))
print("SalePrice Empty\n" + str(data_train["SalePrice"].isnull().sum()))
# 查看SalePrice分布情况
sns.distplot(data_train["SalePrice"])

# 数据清洗
# 填充空值

print(data_train["MSZoning"].head(10))


# plt.show()
