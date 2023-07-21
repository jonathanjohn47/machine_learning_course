import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dataText = """a b c d Class
4.3634 0.46351 1.4281 2.0202 0
3.482 –4.1634 3.5008 –0.07846 0
0.51947 –3.2633 3.0895 –0.98492 0
2.3164 –2.628 3.1529 –0.08622 0
–1.8348 11.0334 3.1863 –4.8888 0
–1.7279 –6.841 8.9494 0.68058 1
–3.3793 –13.7731 17.9274 –2.0323 1
–3.1273 –7.1121 11.3897 –0.08363 1
–2.121 –0.05588 1.949 1.353 1
–1.7697 3.4329 –1.2144 –2.3789 1"""

trialdata = np.array([
    [4.3634, 0.46351, 1.4281, 2.0202, 0],
    [3.482, -4.1634, 3.5008, -0.07846, 0],
    [0.51947, -3.2633, 3.0895, -0.98492, 0],
    [2.3164, -2.628, 3.1529, -0.08622, 0],
    [-1.8348, 11.0334, 3.1863, -4.8888, 0],
    [-1.7279, -6.841, 8.9494, 0.68058, 1],
    [-3.3793, -13.7731, 17.9274, -2.0323, 1],
    [-3.1273, -7.1121, 11.3897, -0.08363, 1],
    [-2.121, -0.05588, 1.949, 1.353, 1],
    [-1.7697, 3.4329, -1.2144, -2.3789, 1]
])


def classify(kernelName):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    if kernelName == 'poly':
        svclassifier = SVC(kernel='poly', degree=8)
        svclassifier.fit(x_train, y_train)
        y_pred = svclassifier.predict(x_test)
        return accuracy_score(y_test, y_pred)
    else:
        svclassifier = SVC(kernel=kernelName)
        svclassifier.fit(x_train, y_train)
        y_pred = svclassifier.predict(x_test)
        return accuracy_score(y_test, y_pred)


trialdata = pd.DataFrame(trialdata, columns=['a', 'b', 'c', 'd', 'Class'])

# SVM algorithm

x = trialdata.drop('Class', axis=1)
y = trialdata['Class']

print("Linear Accuracy:", classify('linear'))
print("Polynomial Accuracy:", classify('poly'))
print("Gaussian Accuracy:", classify('rbf'))
print("Sigmoid Accuracy:", classify('sigmoid'))
