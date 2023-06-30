import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import datasets
from sklearn.mixture import GaussianMixture

# iris = datasets.load_iris()


iris = np.array([
    [random.randint(0,200) for i in range(500)],
    [random.randint(0,200) for i in range(500)],
    [random.randint(0,200) for i in range(500)],
    [random.randint(0,200) for i in range(500)],
])
# X = iris.data[:, :2]
iris = DataFrame(iris)
print(iris)
X = iris.transpose().iloc[:, :2]
d = pd.DataFrame(X)

# plt.scatter(d[0], d[1])
# plt.show()

# Gaussian Mixture Model

gmm = GaussianMixture(n_components=3).fit(d)
labels = gmm.predict(d)

print(labels)
d0 = d[labels == 0]
d1 = d[labels == 1]
d2 = d[labels == 2]

plt.scatter(d0[0], d0[1], c='red')
plt.scatter(d1[0], d1[1], c='green')
plt.scatter(d2[0], d2[1], c='blue')

plt.show()
