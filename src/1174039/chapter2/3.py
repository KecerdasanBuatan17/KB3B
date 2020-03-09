# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:54:20 2020

@author: Liyana
"""

# use one-hot encoding on categorical columns
jatibarang = pd.get_dummies(jatibarang, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])
jatibarang.head()