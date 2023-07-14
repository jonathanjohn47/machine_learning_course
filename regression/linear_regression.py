import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn import linear_model

plt.style.use('ggplot')

# Load the boston dataset
californiaData = fetch_california_housing()

yb = californiaData.target.reshape(-1, 1)
xb = californiaData['data'][:, 0].reshape(-1, 1)

print(pd.DataFrame(californiaData['data']))


plt.scatter(xb, yb)
plt.ylabel('value of house /1000 ($)')
plt.xlabel('number of rooms')

# train dataset
regr = linear_model.LinearRegression()
regr.fit(xb, yb)
plt.plot(xb, regr.predict(xb), color='blue', linewidth=3)
plt.show()
