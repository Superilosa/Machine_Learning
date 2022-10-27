import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
# Gaussian - Float numbers (bell shape), Multinomial (Discrete numbers), Bernoulli (Binary numbers)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv")
y = data['target'].values
X = data.drop('target',axis=1).values
# print(data['target'].value_counts(normalize=True))
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y)
myLR = LogisticRegression(max_iter=100000,n_jobs=-1,C=0.06)
myLR.fit(X_train,y_train)
print("LogisticRegression: "+str(myLR.score(X_test,y_test)))
myG = GaussianNB(priors=[0.55,0.45])
myG.fit(X_train,y_train)
print("GaussianNB: "+str(myG.score(X_test,y_test)))
X = data[["age","sex","cp","slope","thal"]].values
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y)
myM = MultinomialNB()
myM.fit(X_train,y_train)
print("MultinomialNB: "+str(myM.score(X_test,y_test)))
y_pred = myM.predict(X_test)
print(classification_report(y_test,y_pred))