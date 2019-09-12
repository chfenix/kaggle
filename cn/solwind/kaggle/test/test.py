#encoding = utf-8
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series

w = list()
for i in range(10):
    w.append(i)
print(w)
print(type(w))

ser = Series(w)
print(ser)
print(type(ser))