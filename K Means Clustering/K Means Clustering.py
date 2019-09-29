import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

data = make_blobs(n_samples=200, n_features=2, centers=4,
                  cluster_std=1.8, random_state=101)

# print(data)

# Array of samples and columns of data
# print(data[0])
# print(data[0].shape)

# Plotting out fake data to visualize blobs
#plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1])
# plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])

print(kmeans.cluster_centers_)

print(kmeans.labels_)
