import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("https://raw.githubusercontent.com/JangirSumit/kmeans-clustering/master/driver-data.csv")
data.drop("id",axis=1,inplace=True)

cluster = KMeans(4)
cluster.fit(data)
pred = cluster.predict(data)
print(silhouette_score(data,pred))
plt.scatter(data['mean_dist_day'],data['mean_over_speed_perc'],c=pred)
plt.scatter(cluster.cluster_centers_[:,0],cluster.cluster_centers_[:,1],c='red',s=90)
plt.show()