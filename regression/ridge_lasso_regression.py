import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean

# Loading the data
data = pd.read_csv('kc_house_data.csv')

# Drop the non-numerical variables and those with missing values
dropColumns = ['id', 'date', 'sqft_above', 'zipcode']
data = data.drop(dropColumns, axis=1)

# Determine the dependent and independent variables
y = data['price']
X = data.drop('price', axis=1)

# Divide the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Develop a Linear Regression model
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)

# Evaluate the Linear Regression model
print(linearModel.score(X_test, y_test))

# Develop Ridge(L2) Regression Model:
# Estimate different values for lamda
alpha = []
cross_val_scores_ridge = []

# Loop to compute the different scores
for i in range(1, 9):
    ridgeModel = Ridge(alpha=i * 0.25)
    ridgeModel.fit(X_train, y_train)
    scores = cross_val_score(ridgeModel, X, y, cv=10)
    avg_cross_val_score = mean(scores)
    cross_val_scores_ridge.append(avg_cross_val_score)
    alpha.append(i * 0.25)

# Loop to print the different scores
for i in range(0, len(alpha)):
    print(str(alpha[i]) + ' : ' + str(cross_val_scores_ridge[i]))

# the best value of lambda for the data is 2
# Build the Ridge Regression model for the best lambda
ridgeModelChosen = Ridge(alpha=2)
ridgeModelChosen.fit(X_train, y_train)

# Evaluate the Ridge Regression model
print(ridgeModelChosen.score(X_test, y_test))

# Develop Lasso(L1) Regression Model:
# Estimate different values for lambda
lamda = []
cross_val_scores_lasso = []

# Loop to compute the different scores
for i in range(1, 9):
    lassoModel = Lasso(alpha=i * 0.25, tol=0.0925)
    lassoModel.fit(X_train, y_train)
    scores = cross_val_score(lassoModel, X, y, cv=10)
    avg_cross_val_score = mean(scores)
    cross_val_scores_lasso.append(avg_cross_val_score)
    lamda.append(i * 0.25)

# Loop to print the different scores
for i in range(0, len(alpha)):
    print(str(alpha[i]) + ' : ' + str(cross_val_scores_lasso[i]))

# the best value of lambda for the data is 2
# Build the Lasso Regression model for the best lambda
lassoModelChosen = Lasso(alpha=2, tol=0.0925)
lassoModelChosen.fit(X_train, y_train)

# Evaluate the Lasso Regression model
print(lassoModelChosen.score(X_test, y_test))