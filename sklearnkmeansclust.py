#Kmeans for Unsupervised learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(25)
x, y = np.random.randint(0,100, 1000), np.random.randint(0,100, 1000)
# print(x)
# print(y)

df = pd.DataFrame({
    'x':x,
    'y':y
})
# print(df)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(df)
# print(kmeans)

#unlabels dataset to get label  
labels = kmeans.predict(df)
# print(labels)

#centroid state
centroid = kmeans.cluster_centers_
# print(centroid)

#color
color = {1:'red', 2:'blue', 3:'green', 4:'purple', 5:'yellow'}
clr = list(map(lambda x: color[x+1], labels))

plt.figure(figsize=(5,5))
plt.scatter(df['x'], df['y'], c=clr)
for i,v in enumerate(centroid):
    # print(i)
    plt.scatter(*centroid[i], c=color[i+1])


plt.show()
