from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Create GridSearchCV model
param_grid = [{'kernel':['rbf'],'C':[0.001,0.01,0.1,1,10,100],'gamma':[0.001,0.01,0.1,1,10,100]},
              {'kernel':['linear'],'C':[0.001,0.01,0.1,1,10,100]}]
grid_search = GridSearchCV(SVC(),param_grid,cv=6)
# Load data
iris = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=0)
# Find best parameters
grid_search.fit(X_train,y_train)
# Show result scores
print("Test score: ",grid_search.score(X_test,y_test))
print("Best parameters: ",grid_search.best_params_)
print("Validation score: ",grid_search.best_score_)
print("Best estimator: ",grid_search.best_estimator_)
# Show CV results data
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',100)
pd.set_option('display.width',1000)
resData = pd.DataFrame(grid_search.cv_results_)
print(resData.head())