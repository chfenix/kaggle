import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import Series
from sklearn import preprocessing, clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, BayesianRidge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR, LinearSVR

import cn.solwind.kaggle.common.EvalUtil as evalUtil
# from cn.solwind.kaggle.common.AverageWeight import AverageWeight
from cn.solwind.kaggle.common.AverageWeight import AverageWeight

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
target_train = data_train["SalePrice"]
print("======SalePrice Desc======\n", target_train.describe())
print("======SalePrice Empty======\n" + str(target_train.isnull().sum()))
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

    # 众数填充
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
selection_num = 20  # 选取相关度最大的特征数目
valid_feature = corr_mat.nlargest(selection_num + 1, "SalePrice")["SalePrice"].index  # 根据SalePrice截取相关度最大的feNum个特征（不包含SalePrice）
cm = np.corrcoef(data_train[valid_feature].values.T)
sns.set(font_scale=1.25)
# 显示热力图
hm = sns.heatmap(cm, cbar=True, annot=True,
                 square=True, fmt=".2f", annot_kws={"size": 10}, yticklabels=valid_feature.values, xticklabels=valid_feature.values,
                 cmap="YlGnBu")
# plt.show()
valid_feature = valid_feature.delete(0)     # 删除SalePrice本身
print(valid_feature)
valid_feature=['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
# 标准化
scaler = RobustScaler()
data_train_scaled = scaler.fit_transform(data_train[valid_feature])
data_test_scaled = scaler.fit_transform(data_test[valid_feature])

# 降维
pca = PCA(n_components=15)
# data_train_pca = pca.fit_transform(data_train_scaled)    # 降维后数据
# data_test_pca = pca.fit_transform(data_test_scaled)

# 模型评估
print("========= Modeling =========")
# 进行模型交叉验证
models = [LinearRegression(), Ridge(), Lasso(alpha=0.01, max_iter=10000), RandomForestRegressor(n_estimators=400),
          GradientBoostingRegressor(), SVR(), LinearSVR(),
          ElasticNet(alpha=0.001, max_iter=10000), SGDRegressor(max_iter=1000, tol=1e-3), BayesianRidge(),
          KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
          ExtraTreesRegressor()]

# for model in models:
#     evalUtil.model_rmse_log(model,data_train_scaled,target_train)

# 模型调参
# model = RandomForestRegressor()
# grid_search = GridSearchCV(model,{"n_estimators":[100,200,300,400]},cv=5, scoring="neg_mean_squared_log_error")
# grid_search.fit(data_train_scaled,target_train)
# print(grid_search.best_params_, grid_search.best_score_ )

# 模型集成
# Model[RandomForestRegressor]: Score(log)[0.167911]
# Model[GradientBoostingRegressor]: Score(log)[0.153769]
# Model[KernelRidge]: Score(log)[0.174150]
models = [RandomForestRegressor(n_estimators=400), GradientBoostingRegressor(), KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)]
weights = [0.35, 0.5, 0.15]

weight_avg = AverageWeight(models, weights)
evalUtil.model_rmse_log(weight_avg, data_train_scaled, target_train)
# evalUtil.rmse_cv_log(weight_avg,data_train_scaled,target_train)

# 训练模型
execute_model = weight_avg
execute_model.fit(data_train_scaled,target_train)

# 生成提交数据
target_predict = execute_model.predict(data_test_scaled)

prediction = pd.DataFrame(target_predict, columns=['SalePrice'])
result = pd.concat([data_test['Id'], prediction], axis=1)
# 保存预测结果
result.to_csv('./data/Predictions.csv', index=False)