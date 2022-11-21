from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Data
X, y = load_iris(return_X_y=True)
# Pipe
pipe = Pipeline([("classifier", DecisionTreeClassifier())])
# Hybrid
grid_params = [
    {"classifier": [KNeighborsClassifier()], "classifier__n_neighbors": [5, 7, 9, 11],
     "classifier__weights": ["uniform", "distance"]},
    {"classifier": [LogisticRegression(max_iter=1000, multi_class='multinomial', tol=0.000001)],
     "classifier__C": np.logspace(0, 4, 10)},
    {"classifier": [DecisionTreeClassifier()], "classifier__criterion": ["gini", "entropy"],
     "classifier__max_depth": [3, 5, 7, 9]},
    {"classifier": [RandomForestClassifier()], "classifier__n_estimators": [30, 50, 70],
     "classifier__criterion": ["gini", "entropy"]}]
hybrid_model = GridSearchCV(pipe, grid_params, cv=5)

# Count best classifiers
occDict = {"LogisticRegression":0,"KNeighborsClassifier":0,"DecisionTreeClassifier":0,"RandomForestClassifier":0}
for i in range(120):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=i)
    hybrid_model.fit(X_train,y_train)
    occDict[str(hybrid_model.best_estimator_["classifier"])[:str(hybrid_model.best_estimator_["classifier"]).find('(')]] += 1
print(occDict)