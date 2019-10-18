import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import Series
from sklearn import preprocessing, clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR, LinearSVR

import cn.solwind.kaggle.common.EvalUtil as evalUtil
# from cn.solwind.kaggle.common.AverageWeight import AverageWeight
from cn.solwind.kaggle.common.AverageWeight import AverageWeight

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

labelEncoder = preprocessing.LabelEncoder()
oneHotEncoder = preprocessing.OneHotEncoder(sparse=False)

# 读取训练数据
data_train = pd.read_csv("./data/train.csv")
target_train = data_train["Survived"]
print("======Data Train======")
print(data_train.info())

# 读取测试数据
data_test = pd.read_csv("./data/test.csv")
print("======Data Test======")
print(data_test.info())

data_all = pd.concat([data_train,data_test], axis=0)

# 数据分析

# 客舱分为有(1)无(0)两种
data_train.loc[data_train.Cabin.notnull(), "Cabin"] = 1
data_train.loc[data_train.Cabin.isnull(), "Cabin"] = 0

data_test.loc[data_test.Cabin.notnull(), "Cabin"] = 1
data_test.loc[data_test.Cabin.isnull(), "Cabin"] = 0

data_all.loc[data_all.Cabin.notnull(), "Cabin"] = 1
data_all.loc[data_all.Cabin.isnull(), "Cabin"] = 0


# 补全测试集中的票价缺失
# print(data_all.groupby("Pclass").transform(np.mean))
df_fare_mean = data_all[["Pclass","Fare"]].groupby(["Pclass"],as_index=True).mean()     # 根据Pclass计算均值
# print("test",df_fare_mean.loc[data_test.loc[data_test.Fare.isnull()]["Pclass"].values]["Fare"])
# 获取缺失值对应Pclass下的均值进行填充
data_test["Fare"].fillna(df_fare_mean.iloc[data_test.loc[data_test.Fare.isnull()]["Pclass"].values[0]-1]["Fare"],inplace=True)
data_all["Fare"].fillna(df_fare_mean.iloc[data_all.loc[data_all.Fare.isnull()]["Pclass"].values[0]-1]["Fare"],inplace=True)
# print(data_test.info())
# print(data_all.info())


# 预测并补全年龄数据
# 抽取参与年龄计算的字段，使用全部数据进行预测
age_data = data_all[["Age", "Pclass", "SibSp", "Parch", "Fare", "Cabin"]]

age_null_data = age_data.loc[age_data.Age.isnull()]
age_notnull_data = age_data.loc[age_data.Age.notnull()]
X = age_notnull_data.values[:, 1:]
y = age_notnull_data.values[:, 0]
# 使用随机森林预测年龄
rfr = RandomForestRegressor(n_estimators=400)
rfr.fit(X, y)
train_age_predict = rfr.predict(data_train[["Age", "Pclass", "SibSp", "Parch", "Fare", "Cabin"]].loc[data_train.Age.isnull()].values[:, 1:])
data_train.loc[data_train.Age.isnull(), ["Age"]] = train_age_predict
test_age_predict = rfr.predict(data_test[["Age", "Pclass", "SibSp", "Parch", "Fare", "Cabin"]].loc[data_test.Age.isnull()].values[:, 1:])
data_test.loc[data_test.Age.isnull(), ["Age"]] = test_age_predict

# 使用众数补全登船港口
data_train["Embarked"].fillna(data_train["Embarked"].mode().iloc[0], inplace=True)

print("==================Feture Fill End===================")
print(data_train.info())

# 查看不同特性下生存概率是否与特性有关系
plt.figure(figsize=(15,10))
view_feature = ["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked"]
for i, feature_name in enumerate(view_feature):
    plt.subplot(2, 3, (i + 1))
    # 按照属性Groupby，并计算生存均值（由于Survived为0、1，均值即表示生存概率）
    sns.barplot(x=feature_name, y="Survived",
                # hue="Survived",
                data=data_train[[feature_name, "Survived"]].groupby([feature_name], as_index=False).mean())
    # 显示不同属性值下生存与未生还的柱状图组（个人感觉不是太直观）
    # sns.countplot(x=feature_name, hue="Survived", data=data_train)

# 查看年龄和生存率的关系
data_train["Age_int"] = data_train["Age"].astype("int")
plt.subplots(1, 1, figsize=(18, 4))
sns.barplot(x="Age_int", y="Survived",
            data=data_train[["Age_int", "Survived"]].groupby(["Age_int"], as_index=False).mean())
# plt.show()
data_train.drop(["Age_int"],axis=1,inplace=True)

# 特征处理
# 性别oneHotEncoder
# oneHot_sex = pd.DataFrame(oneHotEncoder.fit_transform(data_train["Sex"].values.reshape(-1, 1)),columns=["Sex_1","Sex_2"])
oneHot_sex_train = pd.get_dummies(data_train['Sex'], prefix=data_train[['Sex']].columns[0])
data_train = pd.concat([data_train, oneHot_sex_train], axis=1)
oneHot_sex_test = pd.get_dummies(data_test['Sex'], prefix=data_test[['Sex']].columns[0])
data_test = pd.concat([data_test, oneHot_sex_test], axis=1)
# 港口oneHotEncoder
# oneHot_embarked = pd.DataFrame(oneHotEncoder.fit_transform(data_train["Embarked"].values.reshape(-1, 1)),columns=["Embarked_1","Embarked_2","Embarked_3"])
oneHot_embarked_train = pd.get_dummies(data_train['Embarked'], prefix=data_train[['Embarked']].columns[0])
data_train = pd.concat([data_train, oneHot_embarked_train], axis=1)
oneHot_embarked_test = pd.get_dummies(data_test['Embarked'], prefix=data_test[['Embarked']].columns[0])
data_test = pd.concat([data_test, oneHot_embarked_test], axis=1)

# 特征选择
print("========= Feature Selection =========")
# 计算训练集不同特性相关度 皮尔森相关系数
corr_mat = data_train.corr()
plt.subplots(figsize=(20, 9))
selection_num = 20  # 选取相关度最大的特征数目
valid_feature = corr_mat.nlargest(selection_num + 1, "Survived")["Survived"].index  # 根据Survived截取相关度最大的feNum个特征（不包含Survived）
cm = np.corrcoef(data_train[valid_feature].values.T)
sns.set(font_scale=1.25)
# 显示热力图
hm = sns.heatmap(cm, cbar=True, annot=True,
                 square=True, fmt=".2f", annot_kws={"size": 10}, yticklabels=valid_feature.values, xticklabels=valid_feature.values,
                 cmap="YlGnBu")
# plt.show()
# print(data_train.info())

drop_feature = ["Name", "Sex", "Ticket", "Embarked"]
data_train.drop(drop_feature, axis=1, inplace=True)
data_test.drop(drop_feature, axis=1, inplace=True)

# 标准化
scaler_feature = ["Age","Fare"]
scaler = RobustScaler()
data_train[scaler_feature] = scaler.fit_transform(data_train[scaler_feature])
data_test[scaler_feature] = scaler.fit_transform(data_test[scaler_feature])
print(data_train.head())
print(data_test.head())

# 模型评估
print("========= Modeling =========")
rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=6,
                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
target = data_train["Survived"]
data = data_train.drop(["Survived","PassengerId"],axis=1)


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=None)
rf.fit(X_train,y_train)
y_predict = rf.predict(X_test)

target_predict = rf.predict(data_test.drop(["PassengerId"],axis=1))
print(target_predict)
# 保存预测结果
prediction = pd.DataFrame(target_predict, columns=['Survived'])
result = pd.concat([data_test["PassengerId"], prediction], axis=1)
result.to_csv('./data/Predictions.csv', index=False)