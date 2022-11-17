import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

data = pd.read_csv("https://raw.githubusercontent.com/PacktPublishing/Regression-Analysis-with-R/master/Chapter03/EscapingHydrocarbons.csv",sep=';')
# print(data.head())
# print(data.info())
# print(data.isnull().any())
y = data['AmountEscapingHydrocarbons'].values
X = data.drop('AmountEscapingHydrocarbons',axis=1).values
# print(X.shape)
model = Lasso()
model.fit(X,y)
print(model.score(X,y))

# Pipeline
hybrid = Pipeline(steps=[("Scaler",StandardScaler()),("PCA",PCA(1)),("Algo",Lasso())])
hybrid.fit(X,y)
print(hybrid.score(X,y))
print(np.sum(hybrid.named_steps['PCA'].explained_variance_ratio_))