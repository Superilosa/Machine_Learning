import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def logit(p):
    return np.log(p/(1-p))

x = np.random.uniform(size=10)
print(x)
print(logit(x))
plt.plot(np.arange(10),x,'b')
plt.plot(np.arange(10),logit(x),'g')
plt.show()