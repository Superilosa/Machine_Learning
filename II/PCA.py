import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# კორელაციის მატრიცის ფუნქცია/ფორმულა
def corr_matrix(A):
    return np.dot(A.T,A)/(A.shape[0])
# ვქმნით 6 ერთმანეთთან კორელირებულ (დაკავშირებულ) ცვლადს
t = np.linspace(0,30,15)
x = -3*t+20+np.random.randint(-1,1,15)
x1 = -0.5*x+np.random.randint(-2,2,15)
x2 = 0.5*x+np.random.randint(-3,3,15)
x3 = -0.7*x+np.random.randint(0,1,15)
x4 = 0.9*x+np.random.randint(-5,5,15)
x5 = -0.85*x+np.random.randint(-2,2,15)
Data = np.zeros((15,6))
Data[:,0] = x
Data[:,1] = x1
Data[:,2] = x2
Data[:,3] = x3
Data[:,4] = x4
Data[:,5] = x5
DataStd = StandardScaler().fit_transform(Data)
corr = corr_matrix(DataStd)
print(corr)
w, v = np.linalg.eig(corr)
pca = np.dot(DataStd,v)
# ხელით გამოთვლილი
print('Xelit:')
print(pca[:5,])
# დაიმპორტებული ფუნქციით
print('Sklearn Decomposition:')
pcaF = PCA(n_components=6)
pca = pcaF.fit_transform(DataStd)
print(pca[:5,])