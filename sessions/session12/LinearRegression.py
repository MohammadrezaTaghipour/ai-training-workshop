# Session 12 — Advertising data understanding and MinMax scaling for regression.
# Datasets live in datasets/ next to this file.
# From the repo root: uv run python sessions/session12/LinearRegression.py
#
# pip install pandas matplotlib seaborn scikit-learn joblib
# CRISP
# Business Understanding
# Data Understanding
# Data Preparation
# Modeling
# Evaluation
# Deployment

'''
Supervised:
    y, label, target, outpur
        categorical, Discrete => Classification
        continuous => Regression
            => Linear Regression

unsupervised
    Clustering
'''

# region Business Understanding
'''

'''
# endregion

# region Data Understanding

from pathlib import Path

import pandas as pd

_DATASETS = Path(__file__).resolve().parent / "datasets"

df = pd.read_csv(
    _DATASETS / "Advertising.csv",
    usecols=["Unnamed: 0", "TV", "radio", "newspaper", "sales"],
    index_col="Unnamed: 0",
)

# print(df.columns)
# df.drop(['Unnamed: 0'], axis=1, inplace=True)
# print(df.head().to_string())
# print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns

# sns.heatmap(df.corr())
# plt.show()

# missing data
# print(df.isna().sum())

# print(df.dtypes)
# df.info()
# print(df.describe())

# Data Visualize

# Seaborn, Matplotlib

# Histogram Plot

# sns.histplot(df['sales'])
# plt.show()

# print(df["sales"].median()) # 12.9
# print(df["sales"].skew()) # 0.4075714250767127 => Right Skewed

# outlier data داده های پَرت
# sns.boxplot(x="sales", data=df)
# plt.show()

# sns.pairplot(df)
# plt.show()

# df["TotalSpend"] = df["TV"] + df["radio"] + df["newspaper"]
#
# sns.scatterplot(x="TotalSpend", y="sales", data=df)
# plt.show()
#
# sns.scatterplot(x="TV", y="sales", data=df)
# plt.show()




# endregion


# region Data Preparation ...

# Remove Outlier Data => Data Preparation

# Rescaling

# Normalization, Standardization

X = df.drop('sales', axis=1)
y = df['sales']

# X = df['TV','radio','newspaper']
# y = df['sales']
#
# X = df.iloc[:,:-1]
# y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=0))

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Normalization => MinMaxScaler => Distance => KNN, K-Means, SVM
# Standardization => StandardScaler =>  NN

# 0~ 400,000, 000, 000
# 0~ 10,000

# 0 - 1
# -1~ 1
print(X_train)

min_max_scaler = MinMaxScaler()
X_train_Scaled = min_max_scaler.fit_transform(X_train)
X_test_Scaled = min_max_scaler.transform(X_test)


print('-' * 100)
print(X_train_Scaled) # Normalized
# endregion
