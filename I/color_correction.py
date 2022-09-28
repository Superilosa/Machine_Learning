import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = cv.imread('29037.png')
def color_correction(img,ccm):
    '''
    Input:
        img: H*W*3 numpy array, input image
        ccm: 3*3 numpy array, color correction matrix
    Output:
        output: H*W*3 numpy array, output image after color correction
    '''
    out = np.matmul(img,ccm)
    return out.reshape(img.shape).astype(img.dtype)

ccm = np.array([[1.0234, -0.2969, -0.2266],
                 [-0.5625,  1.6328, -0.0469],
                 [-0.0703,  0.2188,  0.6406]])
out = color_correction(data,ccm)
plt.subplot(1,2,1)
plt.imshow(data)
plt.subplot(1,2,2)
plt.imshow(out)
plt.show()
