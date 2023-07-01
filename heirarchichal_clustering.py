import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as shc

data = DataFrame(np.random.randint(0, 20000, size=(500, 2)))

#data_scaled = DataFrame(normalize(data))

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
plt.show()

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)


plt.scatter(data[0], data[1], c=cluster.labels_, cmap='rainbow')
plt.show()
