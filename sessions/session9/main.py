# pip install pandas matplotlib seaborn missingno
# Session 9 — structured data preparation.
# Datasets live in datasets/ next to this file.
# From the repo root: uv run python sessions/session9/main.py

# cd ...
# .\.venv\Scripts\activate

# pandas
# برای دسترسی به منابع داده ای دوبعدی ساختار جدولی
# و شناخت و پاکسازی داده
# داده ساختار
# Python: List, Tuple, Dictionary, Set
# Pandas: Series, DataFrame
# Series => Column
# DataFrame => Table

from pathlib import Path

import pandas as pd
import numpy as np

_DATASETS = Path(__file__).resolve().parent / "datasets"

# region Pandas - Series _ 1
# print(pd.__version__) #3.0.6

# values = [10, 20, 30, 50, 40, 60, 30, 80, 3, 12]
# print(values)
# print(type(values))
# print(values[0])
# print(values[-1])
# # Mutable, Immutable
# values.sort(reverse=True)
# print(values) #list
# print(values[0])
# values_s = pd.Series(values)
# print('-' * 100)
# print(values_s) #series
# print(values_s[0])

# Bigdata => pySpark

# List => Mutable
# Series => Mutable = YES

# print(values)
# print(values.sort())
# print(values)
#
# print('-' * 100)
#
# values = [10, 20, 30, 50, 40, 60, 30, 80, 3, 12]
# values_s = pd.Series(values)
# print(values_s)
# # values_s.sort_values() # ناپایدار
# # values_s.sort_values(inplace=True) # پایدار
# # values_s = values_s.sort_values()
#
# values_s.sort_values(ascending=False, inplace=True) # نزولی
# # Pandas 3.0
# print(values_s)


# endregion

# region Pandas - Series _ 2

# values = [10, 20, 30, 50, 40,np.nan, 60, 30, 80, 3, 12]
#
# # import statistics as stt
# # print(stt.mean(values))
#
# # Series => Column
#
# values_s = pd.Series(values)
# # print(values_s.keys())
# # print(values_s.values)
# print(values_s.ndim) #  ستون - چند بعدی
# print(values_s.unique()) # Deduplicate
# print(values_s.nunique()) # تعداد مقادیر منحصربفرد
# print(values_s.max()) #
# print(values_s.min())
# print(values_s.mean()) # average
# print(values_s.median()) # میانه  - اول مرتب سازی میکنه اگر فرد باشه عنصر میانی اگر زوج باشه میانگین دوتای وسطی
# print(values_s.count())
# print(values_s.value_counts())
# print(values_s.sum())
# print(values_s.isna().sum()) # تعداد مقادیر خالی رو نشون میده Missing
# print(values_s.isnull().sum()) # تعداد مقادیر خالی رو نشون میده Missing

# endregion

# region Pandas - Series _ 3

# ctrl + alt + L => format
values = [51, 36, 47, 19, 12, 2, 37, 81, 19]
# values_s = pd.Series(values, index=range(1, 10))
# values_s = pd.Series(values, index=range(1, len(values) + 1))
# print(values_s)

# values_s = pd.Series(values, index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
# print(values_s)
# print(values_s[0])
# print(values_s['a'])


# values_dict = {'a': 10, 'b': 50, 'c': 66, 'm': 41, 'n': 22}
# values_s = pd.Series(values_dict)
#
# print(values_s)

# values_1 = [10, 20, 30, 40, 50]
# values_2 = [60, 70, 80, 90, 100]


# values_1_s = pd.Series(values_1)
# values_2_s = pd.Series(values_2)
#
# print(pd.concat([values_1_s, values_2_s]))
# print(pd.concat([values_1_s, values_2_s], ignore_index=True))


# endregion

# region Pandas - DataFrame _ 1 - Basic Methods

# student = {
#     "Name": ["Ali", "Hesam", "Narges", "Mohammad", "Razieh"],
#     "Age": [25, 34, 30, 22, 50],
#     "Education" : ["Master", "PhD", "Bachelors", "PhD", "Master"]
# }
# print(student)
# print('-' * 100)
# from rich import print
# print(student)

#json => rich => print

# df = pd.DataFrame(student)
# print(df)
# print(df.ndim)
# print(df.columns)
# print(df["Name"])
# print(df["Age"])
# print(type(df["Education"]))
# print(df[["Name", "Education"]])

# print(df["Age"].min())
# print(df["Age"].mean())
# print(df["Age"].mode()[0]) # چگال، تراکم، فراوانی
# print(df.info())
# print(df.columns)
# print(df.shape)
# print(df.dtypes)
# print(df.describe())

# df_1 = pd.DataFrame({
#     "Name" : ["Ali", "Hesam", "Narges", "Mohammad"],
#     "Age": [25, 34, 30, 22],
#     "Education" : ["Master", "PhD", "Bachelors", "PhD"]
# })
#
# df_2 = pd.DataFrame({
#     "Name" : ["Ali", "Hesam", "Narges", "Bahar"],
#     "Department" : ["IT", "Sales", "Production", "Sales"],
#     "Education" : ["Master", "PhD", "Bachelors", "PhD"]
# })

# print(pd.concat([df_1, df_2])) #Append
# print(pd.merge(df_1, df_2)) #how="inner"
# print(pd.merge(df_1, df_2, how="left"))
# print(pd.merge(df_1, df_2, how="right"))
# print(pd.merge(df_1, df_2, how="outer", on="Name", suffixes=['_', '_'])) #outer => full

# endregion

# Image, Video, Text, Voice => Data Preparation

# CRISP
    # Business Understanding
    # Data Understanding => Pandas, Matplotlib
    # Data Preparation => Pandas
    # Modeling
    # Evaluation
    # Deployment

#region Pandas - DataFrame - File

# header
# read_csv => csv, txt, unknown => ASCII, Separator, Delimiter = comma ,
# df = pd.read_csv(_DATASETS / "titanic.csv")
# df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv")
# df_2 = pd.read_csv(_DATASETS / "SMSSpamCollection",
#                    header=None,
#                    delimiter='\t')

# print(df)
# print(df.head(5)) # Top 5
# print('-' * 100)
# print(df.tail(5)) # Bottom 5

# print(df.head(1000).to_string())
# print(df.to_string())

# print(df.shape)
# print('-' * 100)
# print(df.columns)
# print('-' * 100)
# print(df.dtypes)
# print('-' * 100)
# print(df.describe().to_string()) # Numerical
# print('-' * 100)
# print(df.isna().sum())
# print('-' * 100)
# print(df.isnull().sum())
# print('-' * 100)
# print(len(df))

# Parquet File => Compress => ColumnStore

# titanic.csv => Size on disk => 59 KB
# DataFrame => Resident in memory => 118.9 KB

# df = pd.read_csv(_DATASETS / "titanic.csv")
# print(df["PassengerId"].dtype)

# int32 => 4 byte
# int64 => 8 byte

# print(df.info())
# print(df.columns)
# '''
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
# '''
#
# axis = {0 => Row , 1 => Column}
# df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1, inplace=True)
# print(df.columns)








#endregion

#region Pandas - DataFrame - Data Preparation

# Fill NA => Not Applicable
# dropna => Null

# titanic_df = pd.read_csv(_DATASETS / "titanic.csv")
# print(titanic_df.isna().sum())
# print(len(titanic_df))
# titanic_df = titanic_df.dropna()
# print(titanic_df.isna().sum())
# print(len(titanic_df))
# عوامل موثر بر نجات یافتن یا غرق شدن در کشتی تایتانیک
# titanic_df = titanic_df.drop("Cabin", axis=1)
# print(titanic_df.columns)
# print(titanic_df.isna().sum())
# Age
    # Mean
    # Median

# Embarked
    # Mode

# mean_age = titanic_df["Age"].mean()
# median_age = titanic_df["Age"].median()
#
# titanic_df["Age"] = titanic_df["Age"].fillna(median_age)
# # titanic_df["Age"].fillna(mean_age, inplace=True)
#
# print(round(titanic_df["Age"].mean(),0))
# # print(titanic_df.isna().sum())
#
# mode_embarked = titanic_df["Embarked"].mode()[0]
#
# titanic_df["Embarked"] = titanic_df["Embarked"].fillna(mode_embarked)








#endregion

#region Pandas - DataFrame - Data Visualization

# pip install matplotlib

import matplotlib.pyplot as plt
import missingno as msno

titanic_df = pd.read_csv(_DATASETS / "titanic.csv")

# msno.matrix(titanic_df)
# plt.show()

# Correlation

import seaborn as sns

print(titanic_df.corr(numeric_only=True).to_string()) # همبستگی مقادیر با هم بصورت دو به دو
# همبستگی:

sns.heatmap(titanic_df.corr(numeric_only=True))
plt.show()



# Dashboard



#endregion



