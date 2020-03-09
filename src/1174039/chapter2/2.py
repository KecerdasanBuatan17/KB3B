# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:41:20 2020

@author: Liyana
"""

# generate binary label (pass/fail) based on G1+G2+G3
# (test grades, each 0-20 pts); threshold for passing is sum>=30
jatibarang['pass'] = jatibarang.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3'])
>= 35 else 0, axis=1)
jatibarang = jatibarang.drop(['G1', 'G2', 'G3'], axis=1)
print(jatibarang.head())