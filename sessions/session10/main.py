# Session 10 — features, target, and a reproducible train/test split.
# Datasets live in datasets/ next to this file.
# From the repo root: uv run python sessions/session10/main.py

# region Description ....

# RAG, Agent
# Pandas
# Git, Github

# pip install pandas matplotlib seaborn missingno
# pip freeze > requirements.txt
# pip install -r requirements.txt
# pip install -r requirements.txt -U

# pip freeze > requirements.txt
# pip freeze >> requirements.txt

# requirements.txt
# نجف زاده
# Cache
# endregion

# region Libraries (Python Package)

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_DATASETS = Path(__file__).resolve().parent / "datasets"

# endregion

# region  dataset split

# Vertical Style
# titanic_df = pd.read_csv(_DATASETS / "titanic.csv",
#                          usecols=['PassengerId', 'Survived', 'Pclass',
#                                   'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
#                          index_col='PassengerId'
#                          )

# print(titanic_df.head(10).to_string())
# print(titanic_df.columns)
# titanic_df =  titanic_df.drop("Name", axis=1)
# titanic_df =  titanic_df.drop("Ticket", axis=1)
# titanic_df = titanic_df.drop(columns=["Name", "Ticket", "Cabin"], axis=1)

# print(titanic_df.columns)
# print(titanic_df.dtypes)
# titanic_df.info()
# print(titanic_df.describe())
# print(titanic_df.isnull().sum()) #Missing Data
# print(titanic_df.isna().sum()) #Missing Data
# print(titanic_df.head(n=20))
# print(titanic_df.tail(n=20))
# print(titanic_df.head().to_string())

# Classification جواب مسئله رو داریم

# dataset => جامعه آماری
# Case, Sample, Instance, Observation

# X => Input, Feature
# y => Output, Target, Label

# Supervised => نظارت شده
# Classification
# Regression

# Unsupervised => بدون نظارت
# Clustering


# Supervised => جواب مسئله را داریم
# with label => y
# y => Discrete {Categorical}, Continuous
# Classification
# y => پیوسته،     گسسته
# Discrete مقادیرش محدود و قابل تفکیک است
# مثل 1,2,3,4
# True, False => 1,0 => Binary
# Classification: Logistic Regression

# y => Continuous
# Regression رگرسیون
# محدوده وسیع تری دارد، مانند قیمت خانه

# X, y
# Problem ???
# Structural
# Feature Selection= Domain Expert, Domain Knowledge,
# Dominate => Feature

# Feature Selection => CNN => Feature Selection

# ------------------ Type 1 -------------------------------------------------
# X = titanic_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]] # Feature, Input
# y = titanic_df["Survived"] # Target, Output, Label

# ------------------ Type 2 -------------------------------------------------
# X = titanic_df.drop('Survived', axis=1) # DataFrame
# y = titanic_df['Survived'] # Series

# ------------------ Type 3 -------------------------------------------------
# X = titanic_df.iloc[:, 1:]
# y = titanic_df.iloc[:, 0]


# print(titanic_df["Age"])
# print(titanic_df[["Age","Survived"]])


# print(titanic_df.loc[:, 'Survived'])
# print(titanic_df.iloc[:, :])
# print(titanic_df.iloc[:, :])

# print(X.head().to_string())
# print(type(X))
#
# print(y)
# print(type(y))

# endregion

# region Query ...

# employee_df = pd.read_csv(_DATASETS / "Employees.csv",
#                           usecols=["Gender", "Start Date", "Last Login Time",
#                                    "Salary", "Bonus %", "Senior Management", "Team"],
#                           parse_dates=["Start Date", "Last Login Time"]
#                           , date_format="%m/%d/%Y"
#                           )
# print(employee_df.columns)
# employee_df.columns = [column_name.replace(" ", "_") for column_name in employee_df.columns]  # List Comprehensive
# print(employee_df.columns)
# employee_df.rename(columns={"Bonus_%": "Bonus", "Team": "Department"}, inplace=True)
# print(employee_df.columns)
# employee_df["Senior Management"] = employee_df["Senior Management"].astype("bool")
# print(employee_df["Team"])
# print(employee_df["Team"].unique())
# print(employee_df["Team"].nunique())
# print(employee_df.head().to_string())
# print(employee_df.dtypes)

# filtered_data = employee_df[employee_df["Salary"].between(60000, 70000)]
# print(filtered_data.head(100).to_string())

# filtered_data = employee_df[employee_df["Start Date"].between('01/01/1995', '12/31/2000')]
# filtered_data = employee_df[employee_df["Last Login Time"].between('08:30 AM', '12:00 AM')]

# print(filtered_data.head(10).to_string())

# print(employee_df.columns)

# endregion

# region random split

# sklearn => pip install scikit-learn
# import sklearn

# Classification, Regression

# prediction => singleton, batch
# model => train, evaluation,deploy => prediction =>

# machine learning: 70% ~ 87% > 90%
# NN, CNN => 99%
# Training: History => TimeSeries

# Solid => Fine Tuning, RAG => Vector Database ===> 1000 => SQL Server 2025, PostgreSQL
# RAG, Agent

titanic_df = pd.read_csv(_DATASETS / "titanic.csv",
                         usecols=["PassengerId", "Survived", "Pclass", "Sex", "Age",
                                  "SibSp", "Parch", "Fare", "Embarked"],
                         index_col="PassengerId")
#
# X = titanic_df.drop("Survived", axis=1)
# y = titanic_df["Survived"]

# X_train, X_test, y_train, y_test = train_test_split(X,
#                                                     y,
#                                                     test_size=0.2,  # TestingSet Size => 20%
#                                                     shuffle=True,
#                                                     random_state=13
#                                                     )
# print(X_train.head(50).to_string()) #DataFrame
# print("-" * 100)
# print(X_test.head(50).to_string())

# Modeling: Machine Learning Algorithms =>


# در تعداد دفعات اجرا نتیجه برای داده های آموزش برابر خواهد بود
# بررسی دقت مدلهای متفاوت بر اساس داده های آموزش و تست یکسان

# Random =>
'''
479               3    male  22.00      0      0    7.5208        S
306               1    male   0.92      1      2  151.5500        S
317               2  female  24.00      1      0   26.0000        S
----------------------------------------------------------------------------------------------------
             Pclass     Sex   Age  SibSp  Parch      Fare Embarked
PassengerId                                                       
710               3    male   NaN      1      1   15.2458        C
440               2    male  31.0      0      0   10.5000        S
841               3    male  20.0      0      0    7.9250        S
'''

# print(len(titanic_df))
# print(titanic_df.shape)

# عددی کردن مقادیر سه روش وجود دارد:
# 1- روش دستی  map, replace
# Pandas 3.0
# print(titanic_df.dtypes)

print(titanic_df["Sex"].nunique())
print(titanic_df["Embarked"].nunique())
print('-' * 100)
# روش دستی جایگزینی

# titanic_df["Sex"] = titanic_df["Sex"].map({"male": 1, "female": 2})
# titanic_df["Embarked"] = titanic_df["Embarked"].map({"C": 2, "S": 1, "Q": 3})

# print(titanic_df.dtypes)

#  get_dummies
# این روش برای زمانی مناسب است که تعداد ستون های رشته ای زیاد است و درجه تنوع داده بالا نیست

print(titanic_df.head(10).to_string())
titanic_df = pd.get_dummies(titanic_df)
print('-' * 100)
print(titanic_df.head(10).to_string())

# endregion
