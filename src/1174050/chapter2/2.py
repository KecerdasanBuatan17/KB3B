# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:36:23 2020

@author: User
"""

# generate binary label (pass/fail) based on G1+G2+G3
# (test grades, each 0-20 pts); threshold for passing is sum>=30
pecel['pass'] = pecelapply(lambda row: 1 if (row['G1']+row['G2']+row['G3'])
>= 35 else 0, axis=1)
pecel = pecel.drop(['G1', 'G2', 'G3'], axis=1)
print(pecel.head())