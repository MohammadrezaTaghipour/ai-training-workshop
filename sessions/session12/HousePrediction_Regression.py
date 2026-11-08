# Session 12 — house price prep: IQR outliers, Age, LabelEncoder, train/test split.
# Datasets live in datasets/ next to this file.
# From the repo root: uv run python sessions/session12/HousePrediction_Regression.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_DATASETS = Path(__file__).resolve().parent / "datasets"

house_df = pd.read_csv(_DATASETS / "HouseNew.csv")
# print(house_df.head())

# sns.boxenplot(x="Price", data=house_df)
# plt.show()

# print(house_df.nlargest(5, 'Price').to_string())

# print(house_df.nsmallest(5, 'Area').to_string())
# print(house_df.nlargest(5, 'Area').to_string())


# sns.boxplot(x='Area', data=house_df)
# plt.show()

Q1 = house_df['Area'].quantile(0.25)
Q3 = house_df['Area'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = house_df[(house_df['Area'] < lower_bound) | (house_df['Area'] > upper_bound)]
print(len(house_df))
print(len(outliers))

house_df_clean = house_df[(house_df['Area'] >= lower_bound) & (house_df['Area'] <= upper_bound)].reset_index(drop=True)


# sns.boxplot(x='Area', data=house_df_clean)
# plt.show()

print(len(house_df_clean))

house_df_clean["Age"] = 1401 -  house_df_clean["YearOfConstruction"]
print(house_df_clean.dtypes)

print(house_df_clean.nlargest(15,"Age").to_string())

print(house_df_clean["Address"].unique())

# Address
# Address => Numeric

# Mapping
# GetDummies
# LabelEncoder ***

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
house_df_clean["Address_code"] = le.fit_transform(house_df_clean["Address"])

# print(house_df_clean[["Area","Age",  "Address_code"]].head(100).to_string())

print(house_df_clean.query("Address =='اباذر' "))
print(house_df_clean.isna().sum())

# house_df_clean["Floor"] = house_df_clean["Floor"].astype("int64")


X = house_df_clean.drop(columns=["Address","Price"])
y = house_df_clean["Price"]

print(X.dtypes)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=42))


from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()










