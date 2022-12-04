import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, f_classif, SelectKBest, chi2

data = pd.read_csv("https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",sep=',')
y = data["Outcome"]
X = data.drop("Outcome",axis=1)

# სვეტების დისპერსიები (სტანდარტული გადახრის კვადრატები):
print(X.var())
# VarianceThreshold-ით ამოვარჩიოთ სვეტები, რომელთა დისპერსია საშუალოზე მეტია
var_med = np.median(X.var().values)
selector = VarianceThreshold(threshold=var_med)
selector.fit(X)
print(X.iloc[:,selector.get_support()].columns)


# mutual_info_classif / f_classif (3 საუკეთესო სვეტი)
def top3col(scores):
    best_cols = []
    for score, col in sorted(zip(scores, X.columns), reverse=True)[:3]:
        best_cols.append(col)
    return best_cols

scores = mutual_info_classif(X,y,random_state=3)
print(top3col(scores))
scores = f_classif(X,y)[0] # [0] is F-statistic values, [1] is P-values
print(top3col(scores))


# SelectKBest with chi2
selector = SelectKBest(score_func=chi2,k=2)
selector.fit(X,y)
print(X.iloc[:,selector.get_support()].columns)