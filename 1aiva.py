import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10,10)
y = 2*x + 1

plt.figure()
plt.plot(x,y)
plt.xlim(-2,10)
plt.ylim(-2,10)

plt.axvline(x=0, color = '#A9A9A9')
plt.show()
