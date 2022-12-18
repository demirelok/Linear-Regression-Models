import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("data/FuelConsumption.csv")
print(df.head())

# The difference between Simple Linear Regression and Multiple Linear Regression is that Simple has one argument whereas Multiple has more than one argument

cdf = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]

msk = np.random.randn(len(df)) <= 0.8
train = df[msk]
test = df[~msk]

regr = linear_model.LinearRegression()
x = np.asanyarray(train[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]])
y = np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit(x, y)
print(f"Coefficients: {regr.coef_}")

y_result = regr.predict(train[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]])
print(y_result)

print("Variance Score: %.2f" % regr.score(x, y))
