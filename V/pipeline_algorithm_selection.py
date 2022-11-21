from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

pipe = Pipeline([("classifier",DecisionTreeClassifier())])

grid_params = [
    {"classifier":[KNeighborsClassifier()],"classifier__n_neighbors":[5,7,9,11],"classifier__weights":["uniform","distance"]},
    {"classifier":[LogisticRegression(max_iter=1000,multi_class='multinomial',tol=0.000001)],"classifier__C":np.logspace(0,4,10)},
    {"classifier":[DecisionTreeClassifier()],"classifier__criterion":["gini","entropy"],"classifier__max_depth":[3,5,7,9]},
    {"classifier":[RandomForestClassifier()],"classifier__n_estimators":[30,50,70],"classifier__criterion":["gini","entropy"]}]
hybrid_model = GridSearchCV(pipe,grid_params,cv=5)
hybrid_model.fit(X_train,y_train)
print("best model is ",hybrid_model.best_estimator_)
print("Parameters ",hybrid_model.best_params_)
print("mean accuracy of the model is ",hybrid_model.score(X_test,y_test))