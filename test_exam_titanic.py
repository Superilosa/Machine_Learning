import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_classif

titanic = pd.read_csv("titanic.csv")
titanic.drop(["PassengerId","Name","Ticket","Cabin","Embarked"],axis=1,inplace=True)
titanic['Sex'] = LabelEncoder().fit_transform(titanic['Sex'])
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())
print(titanic.head())
y = titanic['Survived'].values
X = titanic.drop('Survived',axis=1).values
X = SelectKBest(score_func=f_classif,k=4).fit_transform(X,y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # Already fitted