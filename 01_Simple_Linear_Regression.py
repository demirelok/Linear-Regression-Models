import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import linear_model

df = pd.read_csv("FuelConsumption.csv")
print(df.head())
print(df.shape)

# For simple linear regression in this example, ours will be (EngineSize) and fit (CO2 Emission) independently of one. This does not exist for later uses in the dataset.
# In this context, let's create a "Cumulative Data Set". So let's build a dataset that contains the data we need in our project. As the cumulative name suggests, it means a collapsed clustered data set.

cdf = df[["ENGINESIZE",
          "CO2EMISSIONS"]]  # Instead of bothering to drop columns that we don't need, we select the ones we need and write them to a new df

print(cdf.head(10))

# It's time for a step used in all machine learning algorithms.
# Train Test Splited Data, we will split our dataset into two separate datasets, training and testing. The aim here is to use the train dataset to train our regression model that we will use in this project, and to use the test dataset to test the values we obtained as a result of our model.
# What should we pay attention to when splitting the dataset:
# 1. In this process, we will split 80 percent or at least 70 percent of our dataset into trains. This is because the more different data our model sees, the more successful results it will produce. In other words, if we need to talk for Human, we will train our model with different data with the logic of those who read and travel a lot, so that we can get successful results. The rates given above are best practice. These are the values we learn by the gurus of the business.
# 2. Another thing we need to pay attention to in this process is that both our train and test data sets remain in a homogeneous structure after the split process. For example, let's have data types a, b, c, d in my data set. If I populate my trainset with data type a and run my model for data type b in the future, I will fail. Therefore, when performing the split operation, we have to ensure that all data types are distributed as equally as possible, that is, homogeneous.
# ML (Machine Learning) algorithms

# Split the data

msk = np.random.randn(len(df)) <= 0.8  # we randomly got numbers from 0 to the length of df.

print(
    msk)  # When we examined the result, we saw that an array was returned to us as true false. We filled it as true false thanks to the randn function. Eighty percent of this list is true and twenty percent is false
train = cdf[msk]  # Here 80 percent to train
test = cdf[~msk]  # Here, thanks to the "~" symbol, it means not, so we put the non-hyphens to the test.

print(train)
print(test)

# Note: There are ready-made methods in sklearn for such split operations.



regr = linear_model.LinearRegression()  # we got an instance
train_x = np.asanyarray(train[["ENGINESIZE"]]) # It converts the input given as a parameter in the asanarray() function to an array and delivers it to us. The reason why we do this operation here is to perform a mathematical operation in the function that we will use to calculate the coefficients of the linear line..
train_y = np.asanyarray(train["CO2EMISSIONS"]) # When we send a pandas.Series type value into the asanarray() function, train["CO2EMISSIONS"] gave an error when we send it in this type. Instead, we gave input as train[["CO2EMISSIONS"]] to send in df type.
regr.fit(train_x,train_y)
print("Coefficient: %.2f" % regr.coef_[0])
print("Intercept: %.2f" % regr.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
plt.plot(train_x, regr.coef_[0] * train_x + regr.intercept_, c="r")
plt.title("Scatter Graph")
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

x = float(input("Please type into engine size: "))
y = regr.intercept_ + regr.coef_ * x

print(f"Carbon emission of the vehicle with engine size of {x} : {math.floor(y[0])}")