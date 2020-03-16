# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:16:31 2020

@author: User
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
fig.clf()
ax = fig.gca(projection='3d')
plt.xlabel('MAX')
plt.ylabel('NUM')
plt.zlabel('AVG')
ax.scatter(x, y, z)
ax.set_zlim(0.2, 0.5)
ax.set_xlabel('Max features')
ax.set_ylabel('Num estimator')
ax.set_zlabel('Avg accuracy')
plt.show()
