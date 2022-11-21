from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=14)

# 4 Pipeline:
sc = StandardScaler()
pca = PCA(n_components=2)
pipeline_Knn = Pipeline([('scaler',sc),('pca',pca),('class',KNeighborsClassifier(n_neighbors=5))])
pipeline_lr = Pipeline([('scaler',sc),('pca',pca),('class',LogisticRegression())])
pipeline_dc = Pipeline([('scaler',sc),('pca',pca),('class',DecisionTreeClassifier())])
pipeline_rf = Pipeline([('scaler',sc),('pca',pca),('class',RandomForestClassifier())])

# Pipelines list
pipelines = [pipeline_Knn,pipeline_lr,pipeline_dc,pipeline_rf]
for pipe in pipelines:
    pipe.fit(X_train,y_train)
best_accuracy = 0
best_classifier = 0
best_pipeline = ""
algDict = {0:"KNearestNeighbor",1:"LogisticRegression",2:"DecisionTree",3:"RandomForest"}
for i,model in enumerate(pipelines):
    print("test accuracy for {} model is {}".format(algDict[i],model.score(X_test,y_test)))
    if model.score(X_test,y_test)>best_accuracy:
        best_accuracy=model.score(X_test,y_test)
        best_pipeline = model
        best_classifier = i
print("classifier with best accuracy is: {}".format(algDict[best_classifier]))