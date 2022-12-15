import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# X,_ = make_blobs(n_samples=2000,n_features=2,centers=[[1,5],[8,10]],shuffle=True,random_state=1)
# print(X)
# myKmeans = KMeans(n_clusters=2,max_iter=2000)
# myKmeans.fit(X)
# centers = myKmeans.cluster_centers_
# y_predicted = myKmeans.predict(X)
# plt.scatter(X[:,0],X[:,1],c=y_predicted)
# plt.scatter(centers[:,0],centers[:1],s=90,c='red')
# plt.show()

wine = pd.read_csv("https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv")
wine.drop('Wine',axis=1,inplace=True)
scaler = StandardScaler()
wine = scaler.fit_transform(wine)
tsne = TSNE(n_components=2,perplexity=40,n_iter=2000)
wine_embedding = tsne.fit_transform(wine)
plt.scatter(wine_embedding[:,0],wine_embedding[:,1])
plt.show()