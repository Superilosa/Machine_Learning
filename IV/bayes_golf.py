from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import numpy as np

le = preprocessing.LabelEncoder()
weather= np.array(['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy'])
temp= np.array(['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'])
play= np.array(['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No'])
weatherE = le.fit_transform(weather)
tempE = le.fit_transform(temp)
labels = le.fit_transform(play)
print(np.unique(weather),np.unique(weatherE))
print(np.unique(temp),np.unique(tempE))
print(np.unique(play),np.unique(labels))
features = list(zip(weatherE,tempE))
model = GaussianNB()
model.fit(features,labels)
for i in range(3):
    for j in range(3):
        print("{} {}: {}".format(np.unique(weather)[i],np.unique(temp)[j],model.predict([[i,j]])))
print(model.score(features,labels))