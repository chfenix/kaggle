#encoding = utf-8
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

data_iris = load_boston()
X=data_iris.data
# print(X[:,4])
# print(data_iris.feature_names)
Y=data_iris.target
# print(Y)

data =np.random.randn(7,4)
print(data < 0)