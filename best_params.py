from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import numpy as np

# 2 train_test_split
iris = load_iris()
X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data,iris.target,random_state=7)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval,y_trainval,random_state=4)
best_score = 0
for gamma in [0.001,0.01,0.1,1,10,100]:
    for C in [0.001,0.01,0.1,1,10,100]:
        svm = SVC(gamma=gamma,C=C)
        svm.fit(X_train,y_train)
        score = svm.score(X_valid,y_valid)
        if score > best_score:
            best_score = score
            best_parameters = {'C':C,'gamma':gamma}
svm = SVC(**best_parameters)
svm.fit(X_trainval,y_trainval)
test_score = svm.score(X_test,y_test)
print("Best validation score: ",best_score)
print("Best parameters: ",best_parameters)
print("Final test score: ",test_score)

# cross validation + train_test_split
best_score = 0
for gamma in [0.001,0.01,0.1,1,10,100]:
    for C in [0.001,0.01,0.1,1,10,100]:
        svm = SVC(gamma=gamma,C=C)
        scores = cross_val_score(svm,X_trainval,y_trainval,cv=7)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_parameters = {'C':C,'gamma':gamma}
svm = SVC(**best_parameters)
svm.fit(X_trainval,y_trainval)
test_score = svm.score(X_test,y_test)
print("Best validation score: ",best_score)
print("Best parameters: ",best_parameters)
print("Final test score: ",test_score)

# GridSearchCV
param_grid= {'C':[0.001,0.01,0.1,1,10,100],'gamma':[0.001,0.01,0.1,1,10,100]}
grid_search = GridSearchCV(SVC(),param_grid,cv=8)
grid_search.fit(X_trainval,y_trainval)
test_score = grid_search.score(X_test,y_test)
print("Best validation score: ",grid_search.best_score_)
print("Best parameters: ",grid_search.best_params_)
print("Final test score: ",test_score)
print("Best estimator: ",grid_search.best_estimator_)