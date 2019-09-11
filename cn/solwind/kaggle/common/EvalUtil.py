# 评估工具类
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


# 均方根差(Root Mean Squared Error, RMSE)
def rmse(y_data, y_predict):
    score = np.sqrt(mean_squared_error(y_data, y_predict))
    print("Predict RMSE:[{}]".format(score))
    return score


# 均方根差(Root Mean Squared Error, RMSE)，传入数据取log后计算
def rmse_log(y_data, y_predict):
    score = np.sqrt(mean_squared_error(np.log(y_data), np.log(y_predict)))
    print("Predict RMSE(log):[{}]".format(score))
    return score


# 交叉验证均方根差(cv=10)
def rmse_cv(model, data, target):
    rmse = np.sqrt(-cross_val_score(model, data, target, scoring="neg_mean_squared_error", cv=10))
    print("Model[{}]: Mean[{:.6f}], Std[{:.4f}]".format(model.__class__.__name__, rmse.mean(), rmse.std()))
    return rmse
