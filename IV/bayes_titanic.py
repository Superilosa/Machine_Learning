import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
# ვიშორებთ ზედმეტ სვეტებს
data.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
X = data.drop('Survived',axis=1)
y = data['Survived']
# სქესი გადაგვყავს ბინარულ მაჩვენებელში 1-კაცი, 0-ქალი
mf = pd.get_dummies(X['Sex'])
X = pd.concat([X,mf],axis=1)
X.drop(['Sex','female'],axis=1,inplace=True)
# ცარიელი მნიშვნელობების შევსება (ამ შემთხვევაში საშუალოთი)
# print(X.columns[X.isna().any()]) # ვიგებთ რომელ სვეტში გვხვდება ცარიელი მნიშვნელობები
X['Age'] = X['Age'].fillna(X['Age'].mean())
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=5)
model = GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
# პროგნოზირება
print("*"*50)
print(model.predict(X_test[:10]))
# პროგნოზირების ალბათობები
print("*"*50)
print(model.predict_proba(X_test[:10]))