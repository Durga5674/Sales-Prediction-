# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
Data = pd.read_csv("Advertising.csv")

Data.head(4)
Data.shape
Data.info()
Data.describe().T
Data.isnull().sum()
Data.dtypes
Data.describe().T

# Summary function
def var_summary(x):
    return pd.Series([
        x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(), x.std(), x.var(),
        x.min(), x.quantile(0.01), x.quantile(0.05), x.quantile(0.10), x.quantile(0.25),
        x.quantile(0.50), x.quantile(0.75), x.quantile(0.90), x.quantile(0.95),
        x.quantile(0.99), x.max()
    ], index=[
        'N','NMISS','SUM','MEAN','MEDIAN','STD','VAR','MIN','P1','P5','P10',
        'P25/Q1','P50/Q2','P75/Q3','P90','P95','P99','MAX'
    ])

def var_summary(x):
    uc = x.mean() + (2 * x.std())
    lc = x.mean() - (2 * x.std())
    for i in x:
        if i < lc or i > uc:
            count = 1
        else:
            count = 0
    outlier_flag = count
    return pd.Series([
        x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(), x.std(), x.var(),
        x.min(), x.quantile(0.01), x.quantile(0.05), x.quantile(0.10), x.quantile(0.25),
        x.quantile(0.50), x.quantile(0.75), x.quantile(0.90), x.quantile(0.95),
        x.quantile(0.99), x.max(), lc, uc, outlier_flag
    ], index=[
        'N','NMISS','SUM','MEAN','MEDIAN','STD','VAR','MIN','P1','P5','P10',
        'P25','P50','P75','P90','P95','P99','MAX','LC','UC','outlier_flag'
    ])

Data.apply(lambda x: var_summary(x)).T
var_summary(Data.Newspaper)

# Cleaning
Data["Newspaper"] = Data.Newspaper.clip(lower=10, upper=100)
var_summary(Data.Newspaper)

Data['Sales'] = Data['Sales'].clip(lower=10, upper=100)
var_summary(Data.Sales)

Data['Sales'] = Data['Sales'].fillna(Data['Sales'].mean())

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


# Distribution plots
sns.distplot(Data.Sales)
Data.Sales.hist()
sns.distplot(Data.Newspaper)
Data.Newspaper.hist()
sns.distplot(Data.Radio)
sns.distplot(Data.TV)

# Relationship plots
sns.jointplot(Data.Newspaper)
sns.jointplot(Data.Sales)
sns.jointplot(Data.Sales)
sns.jointplot(Data.TV)
sns.jointplot(Data.Sales)
sns.jointplot(Data.Radio)

# Pairplot and correlations
sns.pairplot(Data)
Data.TV.corr(Data.Sales)
Data.corr()
sns.heatmap(Data.corr())

# Regression Model
import statsmodels.formula.api as smf

import pandas as pd

lm = smf.ols('Sales ~ TV + Radio + Newspaper', Data).fit()
lm.summary()

lm = smf.ols('Sales ~ TV + Radio', Data).fit()
lm.summary()

lm.params
lm.conf_int()

# Metrics
from sklearn import metrics

print(dir(metrics))

lm.rsquared
round(float(lm.rsquared), 2)

# Predictions
ltmpredic = lm.predict(Data)
ltmpredic[1:10]

from sklearn import metrics
print(dir(metrics))

# RMSE
mse = metrics.mean_squared_error(Data.Sales, ltmpredic)
rmse = np.sqrt(mse)
rmse

# Residuals
lm.resid[1:10]
sns.jointplot(Data.Sales)
sns.jointplot(lm.resid)
