# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:12:39 2020

@author: Liyana
"""

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2])
plt.show()