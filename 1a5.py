import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

u = [0,0,1,6]
v = [0,0,4,2]
u_bis = [1,6,v[2],v[3]]
w = [0,0,5,8]
plt.quiver([u[0], u_bis[0], w[0]],
           [u[1], u_bis[1], w[1]],
           [u[2], u_bis[2], w[2]],
           [u[3], u_bis[3], w[3]],
           angles = 'xy', scale_units = 'xy', scale = 1, color = sns.color_palette())

plt.xlim(-2,6)
plt.ylim(-2,9)
plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')
plt.text(-1, 3.5, r'$||\vec{u}||$', color = sns.color_palette()[0], size =20)
plt.text(2.5, 7.5, r'$||\vec{v}||$', color = sns.color_palette()[1], size =20)
plt.text(2, 2, r'$||\vec{u}+\vec{v}||$', color = sns.color_palette()[2], size =20)
plt.show()
