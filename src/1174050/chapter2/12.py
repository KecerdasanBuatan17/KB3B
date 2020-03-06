# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 22:07:50 2020

@author: User
"""

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2])
plt.show()