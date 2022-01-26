import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('income.csv')
# print(df)
k=3
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=k)

labels = kmeans.fit_predict(df[['Age','Income($)']])
# print(labels)
df['cluster'] = labels
# print(df)

# plt.figure(figsize=(10,5))
# plt.scatter(df['Age'], df['Income($)'])
# plt.xlabel('Age')
# plt.ylabel('Income($)')

df1 = df[df['cluster']==0]
# print(df1)
df2 = df[df['cluster']==1]
# print(df2)
df3 = df[df['cluster']==2]
# print(df3)

#as it is not appropriate we need to scale down features
# plt.figure(figsize=(10,5))
# plt.scatter(df1['Age'], df1['Income($)'], c='red', label='Income($)')
# plt.scatter(df2['Age'], df2['Income($)'], c='blue', label='Income($)')
# plt.scatter(df3['Age'], df3['Income($)'], c='green', label='Income($)')
# plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='k', marker='*', label='centroid', s=80)
# plt.legend()

import seaborn as sns
# sns.heatmap(df.corr(), cmap='viridis', annot=True)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
df['Income($)'] = sc.fit_transform(df[['Income($)']])
# print(df)
df['Age'] = sc.fit_transform(df[['Age']])
# print(df)
labels = kmeans.fit_predict(df[['Age','Income($)']])
df['cluster'] = labels
# print(kmeans.inertia_) #sse
# print(df)
# print(kmeans.cluster_centers_)


df1 = df[df['cluster']==0]
# print(df1)
df2 = df[df['cluster']==1]
# print(df2)
df3 = df[df['cluster']==2]
# print(df3)

plt.figure(figsize=(10,6))
plt.scatter(df1['Age'], df1['Income($)'], c='red', label='Income($)', edgecolors='k', alpha=0.5)
plt.scatter(df2['Age'], df2['Income($)'], c='blue', label='Income($)', edgecolors='k', alpha=0.5)
plt.scatter(df3['Age'], df3['Income($)'], c='green', label='Income($)', edgecolors='k', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='k', s=80, marker='*', label='centroid')

plt.legend()

 
k_rang = range(1,10)
sse = []
for k in k_rang:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df[['Age','Income($)']])
    # print(kmeans.inertia_)
    sse.append(kmeans.inertia_)

# print(sse)
# print(sum(sse))

plt.figure(figsize=(10,6))
plt.plot(k_rang, sse)
plt.xlabel('No. of Cluster (k)')
plt.ylabel('SSE')
plt.show()

