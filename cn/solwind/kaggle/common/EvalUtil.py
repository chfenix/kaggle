# 评估工具类
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split


# 均方根差(Root Mean Squared Error, RMSE)
def rmse(y_data, y_predict):
    score = np.sqrt(mean_squared_error(y_data, y_predict))
    print("Predict RMSE:[{:.6f}]".format(score))
    return score


# 均方根差(Root Mean Squared Error, RMSE)，传入数据取log后计算
def rmse_log(y_data, y_predict):
    y_predict[y_predict < 0] = 1e-10
    score = np.sqrt(mean_squared_error(np.log(y_data), np.log(y_predict)))
    print("Predict RMSE(log):[{:.6f}]".format(score))
    return score


# 模型均方根差
def model_rmse(model, data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=None)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    score = rmse(y_test, y_predict)
    print("Model[{}]: Score[{:.6f}]".format(model.__class__.__name__, score))
    return score


# 模型均方根差，传入数据取log后计算
def model_rmse_log(model, data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=None)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    score = rmse_log(y_test, y_predict)
    print("Model[{}]: Score[{:.6f}]".format(model.__class__.__name__, score))
    return score


# 交叉验证均方根差(cv=10)
def rmse_cv(model, data, target):
    rmse = np.sqrt(-cross_val_score(model, data, target, scoring="neg_mean_squared_error", cv=5))
    print("Model[{}]: Mean[{:.6f}], Std[{:.4f}]".format(model.__class__.__name__, rmse.mean(), rmse.std()))
    return rmse
