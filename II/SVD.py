import numpy as np

# A = U*E*V
E = np.diag([4*np.sqrt(5),2*np.sqrt(5)])
V = np.array([[1/np.sqrt(10),-3/np.sqrt(10)],[3/np.sqrt(10),1/np.sqrt(10)]])
U = np.array([[1/np.sqrt(2),-1/np.sqrt(2)],[1/np.sqrt(2),1/np.sqrt(2)]])
print(U.dot(E.dot(V.T)))

# სინგულარული მნიშვნელობის დეკომპოზიცია
A = np.array([[5,5],[-1,7]])
U,E,Vt = np.linalg.svd(A)
E = np.diag(E)
print(U)
print(E)
print(Vt)
print(U.dot(E.dot(Vt)))

# არაკვადრატული მატრიცისთვის (საჭიროა დიაგონალური მატრიცის ზომის შეცვლა)
# A(4,2)=B(4,4)*E(4,2)*Vt(2,2)
# E(2,2) დიაგონალური მატრიცა უნდა გაიზარდოს (4,2) ზომაზე
print('-'*50)
A = np.array([[2,4],[1,3],[0,0],[0,0]])
u, s, v = np.linalg.svd(A)
Sigma = np.zeros((A.shape[0],A.shape[1]))
Sigma[:A.shape[1],:A.shape[1]] = np.diag(s)
print(u)
print(Sigma)
print(v)
print(u.dot(Sigma.dot(v)))