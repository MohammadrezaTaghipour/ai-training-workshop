# # pip install pandas matplotlib seaborn scikit-learn joblib
# # CRISP
# # Business Understanding
# # Data Understanding
# # Data Preparation
# # Modeling
# # Evaluation
# # Deployment => File
#
# '''
# Supervised:
#     y, label, target, outpur
#         categorical, Discrete => Classification
#         continuous => Regression
#             => Linear Regression
#
# unsupervised
#     Clustering
# '''
#
# # region Business Understanding
# '''
#
# '''
# # endregion
#
# # region Data Understanding
#
# import pandas as pd
#
# df = pd.read_csv('datasets/Advertising.csv'
#                  , usecols=['Unnamed: 0', 'TV', 'radio', 'newspaper', 'sales']
#                  , index_col='Unnamed: 0'
#                  )
#
# # print(df.columns)
# # df.drop(['Unnamed: 0'], axis=1, inplace=True)
# # print(df.head().to_string())
# # print(df.corr())
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # sns.heatmap(df.corr())
# # plt.show()
#
# # missing data
# # print(df.isna().sum())
#
# # print(df.dtypes)
# # df.info()
# # print(df.describe())
#
# # Data Visualize
#
# # Seaborn, Matplotlib
#
# # Histogram Plot
#
# # sns.histplot(df['sales'])
# # plt.show()
#
# # print(df["sales"].median()) # 12.9
# # print(df["sales"].skew()) # 0.4075714250767127 => Right Skewed
#
# # outlier data داده های پَرت
# # sns.boxplot(x="sales", data=df)
# # plt.show()
#
# # sns.pairplot(df)
# # plt.show()
#
# # df["TotalSpend"] = df["TV"] + df["radio"] + df["newspaper"]
# #
# # sns.scatterplot(x="TotalSpend", y="sales", data=df)
# # plt.show()
# #
# # sns.scatterplot(x="TV", y="sales", data=df)
# # plt.show()
#
# # endregion
#
#
# # region Data Preparation ...
# # Remove Outlier Data => Data Preparation
# # Rescaling
# # Normalization, Standardization
#
# X = df.drop('sales', axis=1)
# y = df['sales']
#
# # X = df['TV','radio','newspaper']
# # y = df['sales']
# #
# # X = df.iloc[:,:-1]
# # y = df.iloc[:,-1]
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = (
#     train_test_split(X, y, test_size=0.2, random_state=42))
#
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
#
# # Normalization => MinMaxScaler => Distance => KNN, K-Means, SVM
# # Standardization => StandardScaler =>  NN
#
# # 0~ 400,000, 000, 000
# # 0~ 10,000
#
# # 0 - 1
# # -1~ 1
# # print(X_train)
#
# # min_max_scaler = MinMaxScaler()
# # X_train_Scaled = min_max_scaler.fit_transform(X_train)
# # X_test_Scaled = min_max_scaler.transform(X_test)
#
# print(X_train.head(10))
#
# # 76.3,27.5,16.0 => 12
# print('-' * 100)
# # print(X_train_Scaled) # Normalized
#
# from sklearn.linear_model import LinearRegression
#
# model = LinearRegression()
#
# # fit => model train
#
# # model.fit(X_train_Scaled, y_train) # Modeling
# model.fit(X_train.values, y_train) # Modeling
#
#
# # Evaluation ارزیابی
# # Testing Set => Accuracy 20% rescale
# # 65% ~ 90%
# # CNN => TrainingSet, ValidationSet, TestingSet => 99.91 %
# # 50
#
#
#
# # prediction:
# #               Singleton Prediction,
# #               Batch Prediction
#
# y_pred = model.predict(X_test.values)  #Evaluation
# # print(y_pred)
#
# # ERP
#
# # Estimation
#
# from sklearn.metrics import mean_squared_error,mean_absolute_error
# import numpy as np
#
# print(df["sales"].mean()) # 13~14 %
# print('-' * 100)
#
#
# MAE = mean_absolute_error(y_test, y_pred)
# MSE = mean_squared_error(y_test, y_pred)
# RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
#
#
# print('MAE:', round(MAE, 2))
# print('MSE:', round(MSE, 2))
# print('RMSE:', round(RMSE, 2))
#
# # MAE: 1.36
# # MSE: 4.4
# # RMSE: 2.1 => 13%, 14%
#
# # Residual Plot باقیمانده
#
# test_residual = y_test - y_pred
#
# import seaborn as sn
#
# # sn.scatterplot(x= y_test, y= test_residual)
# # plt.axhline(y=0, color='r', linestyle='--')
# # plt.show()
#
# # Pandas => Series, DataFrame
# #           Series => Column => [ ]
# #           DataFrame = Tabular [[    ]]
# # new_data = [[76.3,27.5,16.0]] # 11.63897134
# # new_data = [[17.2,45.9,69.3]] # [12.62381325]
# # new_data = [[8.7,48.9,75]] # [12.82693587]
#
# # print(model.predict(new_data)) #[2670.86210588], [16.7614544]
# # Regression => m, b => y => mx + b
#
# # Final Model:
#
# final_model = LinearRegression()
# final_model.fit(X.values, y)
#
# new_data = [[17.2,45.9,69.3]] # before: [12.62381325], after: [12.30767078]
#
# print(final_model.predict(new_data))
#
# # import sklearn
# # print(sklearn.__version__)
#
# print(final_model.coef_)
# coef = pd.DataFrame(final_model.coef_, X.columns, columns=['Coefficient'])
# print(coef)
#
# # Deployment
# # pip install joblib
#
# # Linear Regression: m, b
#
# from joblib import dump, load
#
# dump(final_model, 'LinearRegression.joblib')
#
#
#
# # endregion


# Session 13 — load the saved Advertising linear model and predict sales.
# Training / evaluation / joblib dump are in the commented regions above.
# Datasets and the .joblib file live next to this script.
# From the repo root:
#   uv run python sessions/session13/LinearRegression.py
#
# Regression
# y = mx + b
# y => sales => ????
# m , b =
# x

from pathlib import Path

from joblib import load

_SESSION = Path(__file__).resolve().parent
_DATASETS = _SESSION / "datasets"

# When uncommenting the training regions above, read:
#   df = pd.read_csv(_DATASETS / "Advertising.csv", ...)
# and dump to: _SESSION / "LinearRegression.joblib"

loaded_model = load(_SESSION / "LinearRegression.joblib")

tv = input("TV: ")
radio = input("Radio: ")
newspaper = input("Newspaper: ")

# Singleton Prediction
result = loaded_model.predict([[float(tv), float(radio), float(newspaper)]])
print(result)
# Model +  NB