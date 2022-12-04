from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, LeaveOneOut, ShuffleSplit
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

# Cross validation
iris = load_iris()
logreg = LogisticRegression(max_iter=2000)
scores = cross_val_score(logreg,iris.data,iris.target,cv=7)
print("Cross-validation scores: ",scores)
print("Average: ",scores.mean())

# StratifiedKFold
data = pd.read_csv("Churn_Modelling.csv")
X = data.iloc[:,3:13]
y = data.iloc[:,13]
geography = pd.get_dummies(X["Geography"],drop_first=True)
gender = pd.get_dummies(X["Gender"],drop_first=True)
X.drop(["Gender","Geography"],axis=1,inplace=True)
X = pd.concat([X,geography,gender],axis=1)
X = X.values
y = y.values
# With for iterator
accuracy = []
skf = StratifiedKFold(n_splits=7)
for train_index,test_index in skf.split(X,y):
    X_train,X_test = X[train_index], X[test_index]
    y_train,y_test = y[train_index], y[test_index]
    logreg.fit(X_train,y_train)
    pred = logreg.predict(X_test)
    score = accuracy_score(pred,y_test)
    accuracy.append(score)
print("StartifiedKFold scores: ",accuracy)
print("Average: ",np.array(accuracy).mean())
# With cross_val_score
score = cross_val_score(logreg,X,y,cv=skf)
print("StratifiedKFold cross-val scores: ",score)
print("Average: ",score.mean())

# KFold with cross_val_score
kfold = KFold(n_splits=7)
score = cross_val_score(logreg,X,y,cv=kfold)
print("KFold cross-val scores: ",score)
print("Average: ",score.mean())
# KFold with shuffle and random state
kfold = KFold(n_splits=7,shuffle=True,random_state=4)
score = cross_val_score(logreg,X,y,cv=kfold)
print("KFold shuffled random cross-val scores: ",score)
print("Average: ",score.mean())

# Leave-one-out
# Will take very long on big datasets
loo = LeaveOneOut()
score = cross_val_score(logreg,iris.data,iris.target,cv=loo)
print("Number of cv iterations: ",len(score))
print("Leave-one-out cross-val scores: ",score)
print("Average: ",score.mean())

# Shuffle-split cross-val score
shuffle_split = ShuffleSplit(test_size=.4,train_size=.6,n_splits=7)
score = cross_val_score(logreg,X,y,cv=shuffle_split)
print("Shuffle-split cross-val scores: ",score)
print("Average: ",score.mean())