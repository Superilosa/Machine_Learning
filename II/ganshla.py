import numpy as np

# საკუთრივი მნიშვნელობების დიაგონალური მატრიცა S და საკუთრივი ვექტორების მატრიცა V
S = np.diag([4,1])
V = np.array([[1,1],[1,-2]])
print(V.dot(S.dot(np.linalg.inv(V))))

# საკუთრივი მნიშნველობები და ვექტორები (eigenvalue,eigenvector)
A = np.array([[3,1],[2,2]])
w,v = np.linalg.eig(A)
print(np.diag(w))
print(v)

# (a * a^T) და (a^T * a) ყოველთვის გვაძლევს სიმეტრიულ მატრიცას
a = np.array([[1,2,3],[4,5,6]])
aaT = np.dot(a,a.T)
aTa = np.dot(a.T,a)
print(aaT)
print(aTa)

# მატრიცის ტრანსპონირებულზე და ტრანსპონირებულის მატრიცაზე ნამრავლის არანულოვანი საკუთრივი მნიშვნელობები ტოლია
a = np.array([[3,1],[2,2]])
aaT = np.dot(a,a.T)
aTa = np.dot(a.T,a)
print('a * aT')
w, v = np.linalg.eig(aaT)
print(np.diag(w))
print(v)
print('aT * a')
w1, v1 = np.linalg.eig(aTa)
print(np.diag(w1))
print(v1)