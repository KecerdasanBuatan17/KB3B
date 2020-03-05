# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:42:26 2020

@author: User
"""

# shuffle rows
pecel = pecel.sample(frac=1)
# split training and testing data
pecel_train = pecel[:500]
pecel_test = pecel[500:]
pecel_train_att = pecel_train.drop(['pass'], axis=1)
pecel_train_pass = pecel_train['pass']
pecel_test_att = pecel_test.drop(['pass'], axis=1)
pecel_test_pass = pecel_test['pass']
pecel_att = pecel.drop(['pass'], axis=1)
pecel_pass = pecel['pass']
# number of passing students in whole dataset:
import numpy as np
print("Passing: %d out of %d (%.2f%%)" % (np.sum(pecel_pass), len(pecel_pass),100*float(np.sum(pecel_pass)) / len(pecel_pass)))