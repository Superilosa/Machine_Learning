import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

x = np.arange(10).reshape(-1,1)
y = np.array([0,0,0,0,1,1,1,1,1,1])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
#create model
model = LogisticRegression(solver='liblinear', C=7, random_state=0)
model.fit(x_train,y_train)
print("Model Intercept is {}".format(model.intercept_))
print("Model coefficient is {}".format(model.coef_))
print("Prediction probabilitys {}".format(model.predict_proba(x_test)))
print("Prediction {}".format(model.predict(x_test)))
print("Score {}".format(model.score(x_test,y_test)))
print("Confusion matrix {}".format(confusion_matrix(y_test,model.predict(x_test))))
print(classification_report(y,model.predict(x)))