import numpy as np
import matplotlib.pyplot as plt

def plotVectors(vecs, cols, alpha =1 ):
    plt.figure()
    plt.axvline(x=0, color = '#A9A9A9', zorder = 0)
    for i in range(len(vecs)):
        x = np.concatenate([[0,0],vecs[i]])
        plt.quiver([x[0]],
               [x[1]],
               [x[2]],
               [x[3]],
               angles='xy', scale_units='xy', scale=1,color=cols[i], alpha=alpha)
orange= '#FF9A13'
blue= '#1190FF'
plotVectors([[1,3],[2,1]],[orange,blue])
plt.xlim(0,5)
plt.ylim(0,5)
plt.show()
