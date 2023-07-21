import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans

import random

Data = {
    'x': [random.randint(0,100) for i in range(500)],
    'y': [random.randint(0,100) for i in range(500)],
}

df = DataFrame(Data, columns=['x', 'y'])

print(df)

kmeans = KMeans(n_clusters=3).fit(df)

plt.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=50)

plt.show()