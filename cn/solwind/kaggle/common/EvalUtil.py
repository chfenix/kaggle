# 评估工具类
import numpy as np
from sklearn import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split


# 均方根差(Root Mean Squared Error, RMSE)
def rmse(y_data, y_predict):
    score = np.sqrt(mean_squared_error(y_data, y_predict))
    print("Predict RMSE:[{:.6f}]".format(score))
    return score


# 均方根差(Root Mean Squared Error, RMSE)，传入数据取log后计算
def rmse_log(y_data, y_predict):
    if np.sum(y_predict < 0) > 0:
        print("Predict Data had less 0! num[{}]".format(np.sum(y_predict < 0)))
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
    print("Model[{}]: Score(log)[{:.6f}]".format(model.__class__.__name__, score))
    return score


# 交叉验证均方根差(cv=10)neg_mean_squared_error
def rmse_cv(model, data, target):
    rmse = np.sqrt(-cross_val_score(model, data, target, scoring="neg_mean_squared_error", cv=10))
    print("Model[{}]: Mean[{:.6f}], Std[{:.4f}]".format(model.__class__.__name__, rmse.mean(), rmse.std()))
    return rmse


# 交叉验证均方根差(cv=10)，neg_mean_squared_log_error
def rmse_cv_log(model, data, target):
    rmse = np.sqrt(-cross_val_score(model, data, target, scoring="neg_mean_squared_log_error", cv=10))
    print("Model[{}]: Log Mean[{:.6f}], Log Std[{:.4f}]".format(model.__class__.__name__, rmse.mean(), rmse.std()))
    return rmse




# 临时使用，未优化
# 使用测试集数据进行RMSE验证
def test_data_valid(model, data, target):
    is_log = 0
    if is_log == 1:
        target = np.log(target)

    test_model = clone(model)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=1)
    test_model.fit(X_train, y_train)
    y_predict = test_model.predict(X_test)

    if (is_log == 1):
        rmse(y_test, y_predict)
    else:
        rmse_log(y_test, y_predict)