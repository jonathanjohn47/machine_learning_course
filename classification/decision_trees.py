from sklearn.preprocessing import LabelEncoder
import pandas as pd
from io import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

dataText = """1 Young high no Fair no
2 Young high no Excellent no
3 Middle high no Fair yes
4 Old medium no Fair yes
5 Old low yes Fair yes
6 Old low yes Excellent no
7 Middle low yes Excellent yes
8 Young medium no Fair no
9 Young low yes Fair yes
10 Old medium yes Fair yes
11 Young medium yes Excellent yes
12 Middle medium no Excellent yes
13 Middle high yes Fair yes
14 Old medium no Excellent no"""

# separate data into lines
lines = dataText.split('\n')

# separate lines into elements
data = [line.split() for line in lines]

# create dataframe
trialdata = pd.DataFrame(data, columns=["""ID age income student credit_rating buys_computer""".split()])

# Drop the 'ID' column as it would not contribute to the model
trialdata.drop('ID', axis=1, inplace=True)

# Instantiate labelencoder object
le = LabelEncoder()

# Apply le on categorical feature columns
trialdata[trialdata.select_dtypes(include=['object']).columns] = trialdata[
    trialdata.select_dtypes(include=['object']).columns].apply(lambda col: le.fit_transform(col))

# separate the data into input and target variables.
x = trialdata.drop('buys_computer', axis=1)
y = trialdata['buys_computer']

print(x)
print(y)

# split the data into training set and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# create the DecisionTreeClassifier.
clf = DecisionTreeClassifier()

# train the model.
clf = clf.fit(x_train, y_train)

# predict the responses for test dataset.
y_pred = clf.predict(x_test)

# find the accuracy score of the model.
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
