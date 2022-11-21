from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV

# pipe-ის შექმნა
pipe=Pipeline(steps=[('scaler',StandardScaler()),('pca',PCA()),('class',DecisionTreeClassifier())])
# პარამეტრების გარეშე
X,y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=4)
pipe.fit(X_train,y_train)
print(pipe.score(X_test,y_test))
# GridSearch (ბადისებრი ძებნის) პარამეტრები
param_grid={'pca__n_components':[10,12,17,20],'class__criterion':['gini','entropy'],'class__max_depth':[5,4,6,7]}
# print(pipe.get_params().keys())
hybrid = GridSearchCV(pipe,param_grid,scoring='accuracy',cv=5)
hybrid.fit(X_train,y_train)
print(hybrid.score(X_test,y_test))
print(hybrid.best_params_)