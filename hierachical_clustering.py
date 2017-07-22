#hierachical clustering decides its own number of clusters. We dont have to descide the no of clusters.

import numpy as np
from sklearn.cluster import MeanShift # as ms
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

centers = [[1,1], [5,5],[10,8]]

# n_samples max value = 10000
X,y = make_blobs(n_samples=500, centers=centers, cluster_std=1)

plt.scatter(X[:,0], X[:,1])

plt.show()
ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters_ = len(np.unique(labels))

print "Numbers of estimated clusters:", n_clusters_

colors = 10*["r.", "g.", "b.", "c.", "k.", "y.", "m."]

print colors
print labels
print cluster_centers

for i in xrange(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

plt.scatter(cluster_centers[:,0], cluster_centers[:,1], marker="X", s=150, linewidths=5, zorder= 10)

plt.show()


