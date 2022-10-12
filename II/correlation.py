import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x = np.linspace(1,20,10)
epsilon = np.random.randint(0,20,10)
k=3;b=17
y = k*x + b + epsilon
print(x)
print(y)
print(np.corrcoef(x,y))
plt.scatter(x,y)
plt.xlabel('x variables')
plt.ylabel('y variables')
plt.show()
# მონაცემების სტანდარტიზაცია
x1 = (x-np.mean(x))/np.std(x)
y1 = (y-np.mean(y))/np.std(y)
data = np.zeros([10,2])
data[:,0] = x1
data[:,1] = y1
corr = np.dot(data.T,data)/(x1.shape[0])
print(corr)

# კორელაციის მატრიცის გამოთვლა ხელით და ფუნქციით
A = np.array([[3,-2,7],[9,4,8]])
A1 = np.zeros((A.shape[0],A.shape[1]))
for i in range(A.shape[0]):
    A1[i] = (A[i]-np.mean(A[i]))/np.std(A[i])
corr = np.dot(A1,A1.T)/(A1.shape[1])
print(corr)
print(np.corrcoef(A))