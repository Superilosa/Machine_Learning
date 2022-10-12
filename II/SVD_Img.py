import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

# Load image in grayscale (cv2.IMREAD_GRAYSCALE=0)
img = cv2.imread('Filosofem.jpg',cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('Filosofem.jpg',0)

# show image
# cv2.imshow('img',img)
# cv2.waitKey(0)

# obtain SVD
U, S, V = np.linalg.svd(img)
print(U.shape,S.shape,V.shape)

# plot images with different number of components
comps = [550,300,100,55,15,3]

plt.figure(figsize= (16,8))
for i in range(len(comps)):
    # @ - matrix multiplication
    low_rank = U[:,:comps[i]] @ np.diag(S[:comps[i]]) @ V[:comps[i],:]
    # low_rank = np.matmul(U[:,:comps[i]],np.matmul(np.diag(S[:comps[i]]),V[:comps[i],:]))
    plt.subplot(2,3,i+1), plt.imshow(low_rank,cmap='gray'), plt.axis('off'),
    plt.title("n components ="+str(comps[i]))
plt.show()