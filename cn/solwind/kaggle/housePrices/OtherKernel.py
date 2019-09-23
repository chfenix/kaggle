from datetime import datetime
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)
from scipy.stats import probplot
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')

SEED = 42
PATH = './data/'

def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set on axis 0
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:1459], all_data.loc[1460:].drop(['SalePrice'], axis=1)

df_train = pd.read_csv(PATH + 'train.csv')
df_test = pd.read_csv(PATH + 'test.csv')
df_all = concat_df(df_train, df_test)

df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set'

dfs = [df_train, df_test]

print('Number of Training Examples = {}'.format(df_train.shape[0]))
print('Number of Test Examples = {}\n'.format(df_test.shape[0]))
print('Training X Shape = {}'.format(df_train.shape))
print('Training y Shape = {}\n'.format(df_train['SalePrice'].shape[0]))
print('Test X Shape = {}'.format(df_test.shape))
print('Test y Shape = {}\n'.format(df_test.shape[0]))

print(df_all.info())
df_all['MSSubClass'] = df_all['MSSubClass'].astype(str)
df_all.sample(5)


def display_missing(df):
    for col in df.columns.tolist():
        if df[col].isnull().sum():
            print('{} column missing values: {}/{}'.format(col, df[col].isnull().sum(), len(df)))
    print('\n')


for df in dfs:
    print('{}'.format(df.name))
    display_missing(df)

# Filling masonry veneer features
df_all['MasVnrArea'] = df_all['MasVnrArea'].fillna(0)
df_all['MasVnrType'] = df_all['MasVnrType'].fillna('None')

# Filling continuous basement features
for feature in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    df_all[feature] = df_all[feature].fillna(0)

# Filling categorical basement features
for feature in ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']:
    df_all[feature] = df_all[feature].fillna('None')

# Filling continuous garage features
for feature in ['GarageArea', 'GarageCars', 'GarageYrBlt']:
    df_all[feature] = df_all[feature].fillna(0)

# Filling categorical garage features
for feature in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    df_all[feature] = df_all[feature].fillna('None')

# Filling other categorical features
for feature in ['Alley', 'Fence', 'FireplaceQu', 'MiscFeature', 'PoolQC']:
    df_all[feature] = df_all[feature].fillna('None')

display_missing(df_all)

# Filling missing values in categorical features with the mode value of neighborhood and house type
for feature in ['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual', 'MSZoning', 'SaleType', 'Utilities']:
    df_all[feature] = df_all.groupby(['Neighborhood', 'MSSubClass'])[feature].apply(lambda x: x.fillna(x.mode()[0]))

# Filling the missing values in LotFrontage with the median of neighborhood
df_all['LotFrontage'] = df_all.groupby(['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))

display_missing(df_all)

print('Training Set SalePrice Skew: {}'.format(df_train['SalePrice'].skew()))
print('Training Set SalePrice Kurtosis: {}'.format(df_train['SalePrice'].kurt()))
print('Training Set SalePrice Mean: {}'.format(df_train['SalePrice'].mean()))
print('Training Set SalePrice Median: {}'.format(df_train['SalePrice'].median()))
print('Training Set SalePrice Max: {}'.format(df_train['SalePrice'].max()))

fig, axs = plt.subplots(nrows=2, figsize=(16, 16))
plt.subplots_adjust(left=None, bottom=5, right=None, top=6, wspace=None, hspace=None)

sns.distplot(df_train['SalePrice'], hist=True, ax=axs[0])
probplot(df_train['SalePrice'], plot=axs[1])

axs[0].set_xlabel('Sale Price', size=12.5, labelpad=12.5)
axs[1].set_xlabel('Theoretical Quantiles', size=12.5, labelpad=12.5)
axs[1].set_ylabel('Ordered Values', size=12.5, labelpad=12.5)

for i in range(2):
    axs[i].tick_params(axis='x', labelsize=12.5)
    axs[i].tick_params(axis='y', labelsize=12.5)

axs[0].set_title('Distribution of Sale Price in Training Set', size=15, y=1.05)
axs[1].set_title('Sale Price Probability Plot', size=15, y=1.05)

plt.show()

df_train, df_test = divide_df(df_all)
# Dropping categorical features
cols = ['GarageYrBlt', 'Id', 'MSSubClass', 'MoSold', 'YearBuilt', 'YearRemodAdd', 'YrSold']

df_train_corr = df_train.drop(cols, axis=1).corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_train_corr.drop(df_train_corr.iloc[1::2].index, inplace=True)
df_train_corr_nd = df_train_corr.drop(df_train_corr[df_train_corr['Correlation Coefficient'] == 1.0].index)

df_test_corr = df_test.drop(cols, axis=1).corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_test_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_test_corr.drop(df_test_corr.iloc[1::2].index, inplace=True)
df_test_corr_nd = df_test_corr.drop(df_test_corr[df_test_corr['Correlation Coefficient'] == 1.0].index)

# Features correlated with target
df_train_corr_nd[df_train_corr_nd['Feature 1'] == 'SalePrice']

# Training set high correlations
df_train_corr_nd.head(10)

fig, axs = plt.subplots(nrows=2, figsize=(50, 50))

sns.heatmap(df_train.drop(cols, axis=1).corr().round(2), ax=axs[0], annot=True, square=True, cmap='coolwarm',
            annot_kws={'size': 12})
sns.heatmap(df_test.drop(cols, axis=1).corr().round(2), ax=axs[1], annot=True, square=True, cmap='coolwarm',
            annot_kws={'size': 12})

for i in range(2):
    axs[i].tick_params(axis='x', labelsize=13)
    axs[i].tick_params(axis='y', labelsize=13)

axs[0].set_title('Training Set Correlations', size=15)
axs[1].set_title('Test Set Correlations', size=15)

plt.show()

num_features = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2',
                'BsmtUnfSF', 'EnclosedPorch', 'GarageArea', 'GarageYrBlt', 'GrLivArea',
                'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal',
                'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF',
                'YearBuilt', 'YearRemodAdd']

fig, axs = plt.subplots(ncols=2, nrows=11, figsize=(12, 80))
plt.subplots_adjust(right=1.5)
cmap = sns.cubehelix_palette(dark=0.3, light=0.8, as_cmap=True)

for i, feature in enumerate(num_features, 1):
    plt.subplot(11, 2, i)
    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', size='SalePrice', palette=cmap, data=df_train)

    plt.xlabel('{}'.format(feature), size=15)
    plt.ylabel('SalePrice', size=15, labelpad=12.5)

    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)

    plt.legend(loc='best', prop={'size': 12})

plt.show()

cat_features = ['Alley', 'BedroomAbvGr', 'BldgType', 'BsmtCond', 'BsmtExposure',
                'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual',
                'CentralAir', 'Condition1', 'Condition2', 'Electrical', 'ExterCond',
                'ExterQual', 'Exterior1st', 'Exterior2nd', 'Fence', 'FireplaceQu',
                'Fireplaces', 'Foundation', 'FullBath', 'Functional', 'GarageCars',
                'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'HalfBath',
                'Heating', 'HeatingQC', 'KitchenAbvGr', 'KitchenQual', 'LandContour',
                'LandSlope', 'LotConfig', 'LotShape', 'MSSubClass', 'MSZoning',
                'MasVnrType', 'MiscFeature', 'MoSold', 'Neighborhood', 'OverallCond',
                'OverallQual', 'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle',
                'SaleCondition', 'SaleType', 'Street', 'TotRmsAbvGrd', 'Utilities', 'YrSold']

fig, axs = plt.subplots(ncols=2, nrows=28, figsize=(18, 120))
plt.subplots_adjust(right=1.5, top=1.5)

for i, feature in enumerate(cat_features, 1):
    plt.subplot(28, 2, i)
    sns.swarmplot(x=feature, y='SalePrice', data=df_train, palette='Set3')

    plt.xlabel('{}'.format(feature), size=25)
    plt.ylabel('SalePrice', size=25, labelpad=15)

    for j in range(2):
        if df_train[feature].value_counts().shape[0] > 10:
            plt.tick_params(axis='x', labelsize=7)
        else:
            plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)

plt.show()

fig, axs = plt.subplots(ncols=2, nrows=11, figsize=(12, 80))
plt.subplots_adjust(right=1.5)

for i, feature in enumerate(num_features, 1):
    plt.subplot(11, 2, i)
    sns.kdeplot(df_train[feature], bw='silverman', label='Training Set', shade=True)
    sns.kdeplot(df_test[feature], bw='silverman', label='Test Set', shade=True)

    plt.xlabel('{}'.format(feature), size=15)

    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)

    plt.legend(loc='best', prop={'size': 15})

plt.show()

df_train['Dataset'] = 'Training Set'
df_test['Dataset'] = 'Test Set'
df_all = concat_df(df_train, df_test)

fig, axs = plt.subplots(ncols=2, nrows=28, figsize=(18, 120))
plt.subplots_adjust(right=1.5, top=1.5)

for i, feature in enumerate(cat_features, 1):
    plt.subplot(28, 2, i)
    sns.countplot(x=feature, hue='Dataset', data=df_all, palette='Set2')

    plt.xlabel('{}'.format(feature), size=25)
    plt.ylabel('Count', size=25)

    for j in range(2):
        if df_train[feature].value_counts().shape[0] > 10:
            plt.tick_params(axis='x', labelsize=7)
        else:
            plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)

    plt.legend(loc='upper right', prop={'size': 15})

plt.show()

sns.set(style='whitegrid')
df_all.head()

df_all['YearBuiltRemod'] = df_all['YearBuilt'] + df_all['YearRemodAdd']
df_all['TotalSF'] = df_all['TotalBsmtSF'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']
df_all['TotalSquareFootage'] = df_all['BsmtFinSF1'] + df_all['BsmtFinSF2'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']
df_all['TotalBath'] = df_all['FullBath'] + (0.5 * df_all['HalfBath']) + df_all['BsmtFullBath'] + (0.5 * df_all['BsmtHalfBath'])
df_all['TotalPorchSF'] = df_all['OpenPorchSF'] + df_all['3SsnPorch'] + df_all['EnclosedPorch'] + df_all['ScreenPorch'] + df_all['WoodDeckSF']
df_all['OverallRating'] = df_all['OverallQual'] + df_all['OverallCond']

df_all['HasPool'] = df_all['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_all['Has2ndFloor'] = df_all['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df_all['HasGarage'] = df_all['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_all['HasBsmt'] = df_all['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_all['HasFireplace'] = df_all['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

df_all['NewHouse'] = 0
idx = df_all[df_all['YrSold'] == df_all['YearBuilt']].index
df_all.loc[idx, 'NewHouse'] = 1

fig = plt.figure(figsize=(12, 6))
cmap = sns.color_palette('Set1', n_colors=10)

sns.scatterplot(x=df_all['GrLivArea'], y='SalePrice', hue='OverallQual', palette=cmap, data=df_all)

plt.xlabel('GrLivArea', size=15)
plt.ylabel('SalePrice', size=15)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)

plt.title('GrLivArea & OverallQual vs SalePrice', size=15, y=1.05)

plt.show()

# Dropping the outliers
df_all.drop(df_all[np.logical_and(df_all['OverallQual'] < 5, df_all['SalePrice'] > 200000)].index, inplace=True)
df_all.drop(df_all[np.logical_and(df_all['GrLivArea'] > 4000, df_all['SalePrice'] < 300000)].index, inplace=True)
df_all.reset_index(drop=True, inplace=True)

bsmtcond_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4}
bsmtexposure_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
bsmtfintype_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
bsmtqual_map = {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
centralair_map = {'Y': 1, 'N': 0}
extercond_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
exterqual_map = {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
fireplacequ_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
functional_map = {'Typ': 0, 'Min1': 1, 'Min2': 1, 'Mod': 2, 'Maj1': 3, 'Maj2': 3, 'Sev': 4}
garagecond_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
garagefinish_map = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
garagequal_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
heatingqc_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
kitchenqual_map = {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
landslope_map = {'Gtl': 1, 'Mod': 2, 'Sev': 3}
lotshape_map = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}
paveddrive_map = {'N': 0, 'P': 1, 'Y': 2}

df_all['BsmtCond'] = df_all['BsmtCond'].map(bsmtcond_map)
df_all['BsmtExposure'] = df_all['BsmtExposure'].map(bsmtexposure_map)
df_all['BsmtFinType1'] = df_all['BsmtFinType1'].map(bsmtfintype_map)
df_all['BsmtFinType2'] = df_all['BsmtFinType2'].map(bsmtfintype_map)
df_all['BsmtQual'] = df_all['BsmtQual'].map(bsmtqual_map)
df_all['CentralAir'] = df_all['CentralAir'].map(centralair_map)
df_all['ExterCond'] = df_all['ExterCond'].map(extercond_map)
df_all['ExterQual'] = df_all['ExterQual'].map(exterqual_map)
df_all['FireplaceQu'] = df_all['FireplaceQu'].map(fireplacequ_map)
df_all['Functional'] = df_all['Functional'].map(functional_map)
df_all['GarageCond'] = df_all['GarageCond'].map(garagecond_map)
df_all['GarageFinish'] = df_all['GarageFinish'].map(garagefinish_map)
df_all['GarageQual'] = df_all['GarageQual'].map(garagequal_map)
df_all['HeatingQC'] = df_all['HeatingQC'].map(heatingqc_map)
df_all['KitchenQual'] = df_all['KitchenQual'].map(kitchenqual_map)
df_all['LandSlope'] = df_all['LandSlope'].map(landslope_map)
df_all['LotShape'] = df_all['LotShape'].map(lotshape_map)
df_all['PavedDrive'] = df_all['PavedDrive'].map(paveddrive_map)

df_all.drop(columns=['Street', 'Utilities', 'PoolQC'], inplace=True)

nominal_features = ['Alley', 'BldgType', 'Condition1', 'Condition2', 'Electrical',
                    'Exterior1st', 'Exterior2nd', 'Fence', 'Foundation', 'GarageType',
                    'Heating', 'HouseStyle', 'LandContour', 'LotConfig', 'MSSubClass',
                    'MSZoning', 'MasVnrType', 'MiscFeature', 'MoSold', 'Neighborhood',
                    'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'YrSold']

encoded_features = []

for feature in nominal_features:
    encoded_df = pd.get_dummies(df_all[feature])
    n = df_all[feature].nunique()
    encoded_df.columns = ['{}_{}'.format(feature, col) for col in encoded_df.columns]
    encoded_features.append(encoded_df)

df_all = pd.concat([df_all, *encoded_features], axis=1)
df_all.drop(columns=nominal_features, inplace=True)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
plt.subplots_adjust(top=1.5, right=1.5)

sns.distplot(df_all['SalePrice'].dropna(), hist=True, ax=axs[0][0])
probplot(df_all['SalePrice'].dropna(), plot=axs[0][1])

df_all['SalePrice'] = np.log1p(df_all['SalePrice'])

sns.distplot(df_all['SalePrice'].dropna(), hist=True, ax=axs[1][0])
probplot(df_all['SalePrice'].dropna(), plot=axs[1][1])

axs[0][0].set_xlabel('Sale Price', size=20, labelpad=12.5)
axs[1][0].set_xlabel('Sale Price', size=20, labelpad=12.5)
axs[0][1].set_xlabel('Theoretical Quantiles', size=20, labelpad=12.5)
axs[0][1].set_ylabel('Ordered Values', size=20)
axs[1][1].set_xlabel('Theoretical Quantiles', size=20, labelpad=12.5)
axs[1][1].set_ylabel('Ordered Values', size=20)

for i in range(2):
    for j in range(2):
        axs[i][j].tick_params(axis='x', labelsize=15)
        axs[i][j].tick_params(axis='y', labelsize=15)

axs[0][0].set_title('Distribution of Sale Price', size=25, y=1.05)
axs[0][1].set_title('Sale Price Probability Plot', size=25, y=1.05)
axs[1][0].set_title('Distribution of Sale Price After log1p Transformation', size=25, y=1.05)
axs[1][1].set_title('Sale Price Probability Plot After log1p Transformation', size=25, y=1.05)

plt.show()

print('Training Set SalePrice Skew: {}'.format(df_all['SalePrice'].skew()))
print('Training Set SalePrice Kurtosis: {}'.format(df_all['SalePrice'].kurt()))

cont_features = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2',
                 'BsmtUnfSF', 'EnclosedPorch', 'GarageArea', 'GrLivArea', 'LotArea',
                 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF',
                 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF']

skewed_features = {feature: df_all[feature].skew() for feature in cont_features if df_all[feature].skew() >= 0.5}
transformed_skews = {}

for feature in skewed_features.keys():
    df_all[feature] = boxcox1p(df_all[feature], boxcox_normmax(df_all[feature] + 1))
    transformed_skews[feature] = df_all[feature].skew()

df_skew = pd.DataFrame(index=skewed_features.keys(), columns=['Skew', 'Skew after boxcox1p'])
df_skew['Skew'] = skewed_features.values()
df_skew['Skew after boxcox1p'] = transformed_skews.values()

fig = plt.figure(figsize=(24, 12))

sns.pointplot(x=df_skew.index, y='Skew', data=df_skew, markers=['o'], linestyles=['-'])
sns.pointplot(x=df_skew.index, y='Skew after boxcox1p', data=df_skew, markers=['x'], linestyles=['--'], color='#bb3f3f')

plt.xlabel('Skewed Features', size=20, labelpad=12.5)
plt.ylabel('Skewness', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=11)
plt.tick_params(axis='y', labelsize=15)

plt.title('Skewed Features Before and After boxcox1p Transformation', size=20)

plt.show()

sparse = []

for feature in df_all.columns:
    counts = df_all[feature].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(df_all) * 100 > 99.94:
        sparse.append(feature)

df_all.drop(columns=sparse, inplace=True)

df_train, df_test = df_all.loc[:1456], df_all.loc[1457:].drop(['SalePrice'], axis=1)
drop_cols = ['Id', 'Dataset']
X_train = df_train.drop(columns=drop_cols + ['SalePrice'])
y_train = df_train['SalePrice']
X_test = df_test.drop(columns=drop_cols)

print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('X_test shape: {}'.format(X_test.shape))


def rmse(y_train, y_pred):
    return np.sqrt(mean_squared_error(y_train, y_pred))


def cv_rmse(model, X=X_train, y=y_train):
    return np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf))


K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=SEED)

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=np.arange(14.5, 15.6, 0.1), cv=kf))
lasso = make_pipeline(RobustScaler(), LassoCV(alphas=np.arange(0.0001, 0.0009, 0.0001), random_state=SEED, cv=kf))
elasticnet = make_pipeline(RobustScaler(),
                           ElasticNetCV(alphas=np.arange(0.0001, 0.0008, 0.0001), l1_ratio=np.arange(0.8, 1, 0.025),
                                        cv=kf))
svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01, max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=SEED)
lgbmr = LGBMRegressor(objective='regression',
                      num_leaves=4,
                      learning_rate=0.01,
                      n_estimators=5000,
                      max_bin=200,
                      bagging_fraction=0.75,
                      bagging_freq=5,
                      bagging_seed=SEED,
                      feature_fraction=0.2,
                      feature_fraction_seed=SEED,
                      verbose=0)
xgbr = XGBRegressor(learning_rate=0.01,
                    n_estimators=3500,
                    max_depth=3,
                    gamma=0.001,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    objective='reg:linear',
                    nthread=-1,
                    seed=SEED,
                    reg_alpha=0.0001)
stack = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, svr, gbr, lgbmr, xgbr), meta_regressor=xgbr,
                            use_features_in_secondary=True)

models = {'RidgeCV': ridge,
          'LassoCV': lasso,
          'ElasticNetCV': elasticnet,
          'SupportVectorRegressor': svr,
          'GradientBoostingRegressor': gbr,
          'LightGBMRegressor': lgbmr,
          'XGBoostRegressor': xgbr,
          'StackingCVRegressor': stack}
predictions = {}
scores = {}

for name, model in models.items():
    start = datetime.now()
    print('[{}] Running {}'.format(start, name))

    model.fit(X_train, y_train)
    predictions[name] = np.expm1(model.predict(X_train))

    score = cv_rmse(model, X=X_train, y=y_train)
    scores[name] = (score.mean(), score.std())

    end = datetime.now()

    print('[{}] Finished Running {} in {:.2f}s'.format(end, name, (end - start).total_seconds()))
    print('[{}] {} Mean RMSE: {:.6f} / Std: {:.6f}\n'.format(datetime.now(), name, scores[name][0], scores[name][1]))

def blend_predict(X):
    return ((0.1 * elasticnet.predict(X)) +
            (0.05 * lasso.predict(X)) +
            (0.1 * ridge.predict(X)) +
            (0.1 * svr.predict(X)) +
            (0.1 * gbr.predict(X)) +
            (0.15 * xgbr.predict(X)) +
            (0.1 * lgbmr.predict(X)) +
            (0.3 * stack.predict(X)))

blended_score = rmse(y_train, blend_predict(X_train))
print('Blended Prediction RMSE: {}'.format(blended_score))

fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(18, 36))
plt.subplots_adjust(top=1.5, right=1.5)

for i, model in enumerate(models, 1):
    plt.subplot(4, 2, i)
    plt.scatter(predictions[model], np.expm1(y_train))
    plt.plot([0, 800000], [0, 800000], '--r')

    plt.xlabel('{} Predictions (y_pred)'.format(model), size=20)
    plt.ylabel('Real Values (y_train)', size=20)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    plt.title('{} Predictions vs Real Values'.format(model), size=25)
    plt.text(0, 700000, 'Mean RMSE: {:.6f} / Std: {:.6f}'.format(scores[model][0], scores[model][1]), fontsize=25)

plt.show()

scores['Blender'] = (blended_score, 0)

fig = plt.figure(figsize=(24, 12))

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.xlabel('Model', size=20, labelpad=12.5)
plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=11)
plt.tick_params(axis='y', labelsize=12.5)

plt.title('Scores of Models', size=20)

plt.show()

submission_df = pd.DataFrame(columns=['Id', 'SalePrice'])
submission_df['Id'] = df_test['Id']
submission_df['SalePrice'] = np.expm1(blend_predict(X_test))
submission_df.to_csv('submissions.csv', header=True, index=False)
submission_df.head(10)
