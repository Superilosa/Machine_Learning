from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# სხვადასხვა n_neighbor-ის მიხედვით გენერირებული მონაცემების ტესტირება
X, y = make_classification(n_samples=40,n_features=5,n_informative=2,n_redundant=3,n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=8)
k_range = range(1,20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
plt.plot(k_range,scores,'g-d')
plt.show()
# კორელაციის მატრიცა
X_std = StandardScaler().fit_transform(X)
X_std = X_std.T
print(X_std.shape)
corr_std = np.corrcoef(X_std)
print(corr_std.shape)
print(corr_std)