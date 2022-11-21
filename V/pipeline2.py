from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,MaxAbsScaler,LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data',sep='\s+',header=None)
le = LabelEncoder()
data.iloc[:,8] = le.fit_transform(data.iloc[:,8])
X = data.iloc[:,1:8]
y = data.iloc[:,8]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
pipe = Pipeline(steps=[('scaler',StandardScaler()),('selector',VarianceThreshold()),('classifier',RandomForestClassifier())])
params = {'scaler':[StandardScaler(),MinMaxScaler(),MaxAbsScaler(),Normalizer()],'selector__threshold':[0,0.001,0.01],'classifier__n_estimators':[50,60,90,100,120]}
hybrid = GridSearchCV(pipe,params,scoring='accuracy',cv=2)
hybrid.fit(X_train,y_train)
print(hybrid.best_params_,hybrid.best_score_)
print("Training set score: "+str(hybrid.score(X_train,y_train)))
print("Test set score: "+str(hybrid.score(X_test,y_test)))