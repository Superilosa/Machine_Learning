import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10,10,0.0001)
# y=x**2
y = x**2
plt.figure(figsize=(12,12))
plt.subplot(3,3,1)
plt.plot(x,y)
plt.title("x**2")
# y = x**2+3*x+10
y = x**2+3*x+10
plt.subplot(3,3,2)
plt.plot(x,y)
plt.title("x**2+3*x+10")
# წრფე გაივლის (1,4) და (2,6) წერტილებში
a = [1,2]
b = [4,6]
plt.subplot(3,3,3)
plt.plot(a,b)
plt.title("წრფე რომელიც გაივლის (1,4) და (2,6) წერტილებში")
# მხების განტოლება
x = np.arange(-20,20,0.0001)
y = -4*x**2 + 2*x + 10
plt.subplot(3,3,4)
plt.plot(x,y)
ytangent = 82*x + 410
plt.plot(x,ytangent)
plt.title("-4*x**2+2*x+10 და მისი მხები -10 წერტილში")
# ექსოინენციური ფუნქცია e**x
x = np.arange(-4,4,0.0001)
y = np.exp(x)
plt.subplot(3,3,5)
plt.plot(x,y)
plt.title("ექსპონენციური ფუნქცია")
# სიგმოიდი
y = np.exp(x)/(np.exp(x)+1)
plt.subplot(3,3,6)
plt.plot(x,y)
plt.title("სიგმოიდი")
# tanhx
y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
plt.subplot(3,3,7)
plt.plot(x,y)
plt.title("tanhx")
#r relu (rectified linear unit)
def rectified(x):
    return max(0,x)
x = np.arange(-10,11)
print(x)
y = [rectified(i) for i in x]
plt.subplot(3,3,8)
plt.plot(x,y)
plt.title("Relu(x)")

plt.show()