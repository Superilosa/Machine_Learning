import numpy as np
import cv2 as cv

a = np.array([[0,-7],[2,9]])
b = np.array([[-8,8],[0,6]])

# პირველი სვეტის ბეჭდვა
print(a[:,0])
# მეორე სვეტის ბეჭდვა
print(a[:,1])
# პირველი რიგის ბეჭდვა
print(a[0,:])
# მეორე რიგის ბეჭდვა
print(a[1,:])
# დეტერმინანტი
print(np.linalg.det(a))
# მთავარი დიაგონალის წევრთა ჯამი
print(np.trace(a))
# ჯამი
print(a+b)

# სურათების მატრიცა
# img1 = cv.imread('logos.jpg')
# img2 = cv.imread('logo2.jpg')
# img2 = cv.resize(img2,(img1.shape[0],img1.shape[1]))
# img1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
# img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
# cv.imshow('logos',img1)
# cv.imshow('logo 2',img2)
# cv.imshow('addition',img1+img2)
# cv.imshow('weighted addition',cv.addWeighted(img1,0.2,img2,0.8,0))
# alpha = 0.8
# aWeightResult = alpha*img1 + (1-alpha)*img2
# aWeightResult = np.around(aWeightResult).astype(np.uint8)
# cv.imshow('Alpha weighted addition',aWeightResult)
# cv.waitKey(0)

# ერთეულოვანი მატრიცა
i = np.identity(3,int)
# დიაგონალური მატრიცა
d = np.diag([4,-3,0,19])
print(i);print(d)
# გამრავლება
print(np.dot(a,b))
# შებრუნებული მატრიცა
a1 = np.linalg.inv(a)
print(a1)
print(np.dot(a,a1))
# განტოლება a*x=b -> x=a1*b
print(np.dot(a1,b))
print(np.linalg.solve(a,b))
# ტრანსპონირება
print(a.T)
print(np.transpose(a))
# რანგი
print(np.linalg.matrix_rank(a))

# ვექტორის პროექცია მეორეზე
