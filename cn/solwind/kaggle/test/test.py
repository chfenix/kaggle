#encoding = utf-8
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series

import seaborn as sns
import matplotlib.pyplot as plt
titanic = sns.load_dataset("titanic")
sns.countplot(x="sex", hue="survived", data=titanic)
plt.show()