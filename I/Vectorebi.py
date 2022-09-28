import numpy as np
import scipy
import math
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# # ვექტორები
# a = np.array([9,4,-1])
# b = np.array([0,3,-1])
#
# # ჯამი
# print(a+b)
# print(a-b)
# # სკალარული ნამრავლი
# print(np.dot(a,b))
# # სიგრძე (L**2)
# print(np.linalg.norm(b))
# # (L**1)
# print(np.linalg.norm(b,1))
# # (L**0)
# print(np.linalg.norm(b,0))
# print(np.count_nonzero(b))
# # (L**inf)
# print(np.linalg.norm(b,np.inf))
# # მანძილი ვექტორებს შორის
# print(np.linalg.norm(a-b))
# print(np.sqrt(np.sum((a-b)**2)))
# print(scipy.spatial.distance.euclidean(a,b))
# # ვექტორებს შორის კუთხის კოსინუსი
# cos = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
# print(cos)
# print(math.degrees(math.acos(cos)))
# print(np.rad2deg(np.arccos(cos)))
# print(cosine_similarity(a.reshape(1,3),b.reshape(1,3))[0][0])

# ვექტორების აგება გრაფიკულად
def plotVectors(vecs,cols):
    plt.axvline(color='#A9A9A9',zorder=0)
    plt.axhline(color='#A9A9A9',zorder=0)
    for i in range(len(vecs)):
        plt.quiver(0,0,vecs[i][0],vecs[i][1],angles='xy',scale_units='xy',scale=1,color=cols[i])
        plt.axis([-1,10,-10,10])

u = np.array([0,-7])
v = np.array([2,9])
orange = '#FF9A13'
blue = '#1190FF'
plotVectors([u,v],cols=[orange,blue])
plt.show()