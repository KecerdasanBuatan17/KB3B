# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:53:04 2020

@author: Alit
"""
import numpy as np
np.set_printoptions(precision=2) 
plt.figure(figsize=(60,60), dpi=300) 
plot_confusion_matrix(cm, classes=birds, normalize=True) 
plt.show()
