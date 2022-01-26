#Unspervised Learning
#libraries used for Kmeans clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(26)
x = np.random.randint(0,100, 1000)
y = np.random.randint(0,100, 1000)
# print(x)
# print(y)

#unlabel dataset => kmeans clustering used for unlabel dataset
df = pd.DataFrame({
    'x':x,
    'y':y
})
# print(df)

# no. of clustering 
k=5

#centroid as per the no. of clustering
centroid = {
    i+1: [np.random.randint(0,100), np.random.randint(0,100)] for i in range(k)
}
# print(centroid)

#as per no. of clustering color are assigned ie 5
color = {1:'red', 2:'blue', 3:'green', 4:'cyan', 5:'hotpink'}

#plotting for visualization
plt.figure(figsize=(5,5))
plt.scatter(df['x'], df['y'], c='k')
for i in centroid.keys():
    plt.scatter(*centroid[i], c=color[i])

#assignment state
def assignment(df, centroid):
    for i in centroid.keys():
        df['distance_from_{}'.format(i)] = np.sqrt(
            (df['x'] - centroid[i][0])**2 +
            (df['y'] - centroid[i][1])**2
        )
    #     df['sse_x_{}'.format(i)] = (df['x'] - centroid[i][0])**2
    #     df['sse_y_{}'.format(i)] = (df['y'] - centroid[i][1])**2
    # sse_x = ['sse_x_{}'.format(i) for i in centroid.keys()]
    # sse_y = ['sse_y_{}'.format(i) for i in centroid.keys()]
    centroid_distance = ['distance_from_{}'.format(i) for i in centroid.keys()]
    # print(centroid_distance)
    df['closest'] = df.loc[:, centroid_distance].idxmin(axis=1).map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: color[x])

    # df['sse_x'] = df.loc[:,sse_x].sum(axis=1)
    # df['sse_y'] = df.loc[:,sse_y].sum(axis=1)
    # df['sse'] = df.loc[:,['sse_x', 'sse_y']].sum(axis=1)

    return df

df = assignment(df, centroid)
# print(df)

plt.figure(figsize=(5,5))
plt.scatter(df['x'], df['y'], c=df['color'], edgecolors='k', alpha=0.2)
for i in centroid.keys():
    plt.scatter(*centroid[i], c=color[i])

#copying for plot marking previous to current centroid
import copy 
old_centroid = copy.deepcopy(centroid)
# print(old_centroid)

#updating the centroid 
def update(centroid):
    for i in centroid.keys():
        centroid[i][0] = np.mean(df[df['closest']==i]['x'])
        centroid[i][1] = np.mean(df[df['closest']==i]['y'])
    return centroid

centroid = update(centroid)
# print(centroid)
# print(old_centroid)
df = assignment(df, centroid)

plt.figure(figsize=(5,5))
ax = plt.axes()
plt.scatter(df['x'], df['y'], c=df['color'], edgecolors='k', alpha=0.2)
for i in centroid.keys():
    plt.scatter(*centroid[i], c=color[i])

# plot marking previous to current centroid
for i in old_centroid.keys():
    x = old_centroid[i][0]
    y = old_centroid[i][1]
    dx = (centroid[i][0] - old_centroid[i][0]) * 0.75
    dy = (centroid[i][1] - old_centroid[i][1]) * 0.75

    ax.arrow(x, y, dx, dy, head_width=2, head_length=3, fc=color[i], ec=color[i])

plt.figure(figsize=(5,5))
plt.scatter(df['x'], df['y'], c=df['color'], edgecolors='k', alpha=0.2)
for i in centroid.keys():
    plt.scatter(*centroid[i], c=color[i])

# after validation of centroid, then assigning the df then checking previous closest_distance equal the current closest_distance then it will break  
while True:
    closest_distance = copy.deepcopy(df['closest'])
    centroid = update(centroid)
    df = assignment(df, centroid)
    if closest_distance.equals(df['closest']):
        break

plt.figure(figsize=(5,5))
plt.scatter(df['x'], df['y'], c=df['color'], edgecolors='k', alpha=0.2)
for i in centroid.keys():
    plt.scatter(*centroid[i], c=color[i])




# plt.show()

# print(df)