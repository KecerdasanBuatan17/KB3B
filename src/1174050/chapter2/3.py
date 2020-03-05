# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:41:22 2020

@author: User
"""

# use one-hot encoding on categorical columns
pecel = pd.get_dummies(pecel, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])
pecel.head()