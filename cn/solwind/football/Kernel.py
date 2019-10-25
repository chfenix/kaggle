import tensorflow as tf
import numpy as np
import time
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 读取结果数据
data_result = pd.read_csv("./data/ENG_PR/result.csv")
print("data_result:", data_result.head())

# 读取训练数据
data_odds = pd.read_csv("./data/ENG_PR/odds.csv")
print(data_odds.head())
data_odds = data_odds[data_odds["Type"] == "O"].groupby("Id", as_index=False)["WinsOdds", "DrawsOdds", "LossesOdds"].mean()
print("data_odds:", data_odds.head())

# 组合数据
data_all = pd.merge(data_result, data_odds)
data_all.loc[data_all["Wins"] == 1, "Result"] = 1   # 胜
data_all.loc[data_all["Draws"] == 1, "Result"] = 2  # 平
data_all.loc[data_all["Losses"] == 1, "Result"] = 3 # 负

# 赔率保留2位小数便于观察分布
data_all["WinsOdds"] = round(data_all["WinsOdds"], 2)
data_all["DrawsOdds"] = round(data_all["DrawsOdds"], 2)
data_all["LossesOdds"] = round(data_all["LossesOdds"], 2)
print(data_all)
print(data_all.info())

print("===========Win Odds describe=============")
print(data_all[data_all["Wins"] == 1]["WinsOdds"].describe())
print(data_all[data_all["Wins"] == 1]["DrawsOdds"].describe())
print(data_all[data_all["Wins"] == 1]["LossesOdds"].describe())

print("===========Draws Odds describe=============")
print(data_all[data_all["Draws"] == 1]["WinsOdds"].describe())
print(data_all[data_all["Draws"] == 1]["DrawsOdds"].describe())
print(data_all[data_all["Draws"] == 1]["LossesOdds"].describe())

print("===========Losses Odds describe=============")
print(data_all[data_all["Losses"] == 1]["WinsOdds"].describe())
print(data_all[data_all["Losses"] == 1]["DrawsOdds"].describe())
print(data_all[data_all["Losses"] == 1]["LossesOdds"].describe())

# plt.figure(figsize=(15,10))
# sns.scatterplot(x="WinsOdds", y="LossesOdds", hue="Wins", data=data_all)
# sns.boxplot(x="Result", y="WinsOdds", data=data_all, palette="Set3")
corr_mat = data_all.corr()
plt.subplots(figsize=(20, 9))
selection_num = 20  # 选取相关度最大的特征数目
valid_feature = ["Wins","Draws","Losses","WinsOdds","DrawsOdds","LossesOdds","Result"]
cm = np.corrcoef(data_all[valid_feature].values.T)
sns.set(font_scale=1.25)
# 显示热力图
hm = sns.heatmap(cm, cbar=True, annot=True,
                 square=True, fmt=".2f", annot_kws={"size": 10}, yticklabels=valid_feature, xticklabels=valid_feature,
                 cmap="YlGnBu")

# plt.show()