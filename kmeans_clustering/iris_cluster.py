import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans


iris_data = datasets.load_iris()
X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = iris_data.target
X.drop(['sepal length (cm)','sepal width (cm)'], axis=1, inplace=True)
# print(X, y)

plt.figure(figsize=(10,5))
plt.scatter(X['petal length (cm)'], X['petal width (cm)'])
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')

#no. of cluster
k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit_transform(X)
labels = kmeans.fit_predict(X)
# print(labels)
X['cluster'] = labels

X0 = X[X['cluster']==0]
X1 = X[X['cluster']==1]
X2 = X[X['cluster']==2]

plt.figure(figsize=(10,5))
plt.scatter(X0['petal length (cm)'], X0['petal width (cm)'], c='red', edgecolors='k', alpha=0.5, label='petal width (cm)')
plt.scatter(X1['petal length (cm)'], X1['petal width (cm)'], c='blue', edgecolors='k', alpha=0.5, label='petal width (cm)')
plt.scatter(X2['petal length (cm)'], X2['petal width (cm)'], c='green', edgecolors='k', alpha=0.5, label='petal width (cm)')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='k', marker='*', s=80, label='centroid')


rang_k = range(1,10)
sse = []

for k in rang_k:
    kmeans =  KMeans(n_clusters=k)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(5,5))
plt.plot(rang_k, sse)
plt.xlabel('no. of cluster / k')
plt.ylabel('Sum of Squared Error / SSE')

plt.show()