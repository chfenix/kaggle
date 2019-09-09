#encoding = utf-8
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

data_iris = load_iris()
X=data_iris.data
print(X)
print(data_iris.feature_names)
Y=data_iris.target
print(Y)


knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, Y, cv=10, scoring='neg_mean_squared_error')
print(scores)

print('sqrt计算各个元素的平方根：')
num = np.random.randint(1,7,size = (2,3))
print(num)
print(np.sqrt(num))
print()