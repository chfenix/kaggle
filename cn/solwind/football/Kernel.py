import tensorflow as tf
import numpy as np
import time
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 读取结果数据
data_result = pd.read_csv("./data/ENG_PR/result.csv")
print("data_result:", data_result.head())

# 读取训练数据
data_odds = pd.read_csv("./data/ENG_PR/odds_format.csv")
data_odds = data_odds.groupby("Id", as_index=False)["Wins", "Draws", "Losses"].mean()
print("data_odds:", data_odds.head())

# 组合数据
data_all = pd.merge(data_result, data_odds)
print(data_all.head())
