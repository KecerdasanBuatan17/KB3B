# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:40:39 2020

@author: User
"""
#%%
# load dataset (student mat pakenya)
import pandas as pd
pisang = pd.read_csv('student-mat.csv', sep=';')
print(len(pisang))
#%%

#%%
pisang['pass'] = pisang.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3'])
>= 35 else 0, axis=1)
pisang = pisang.drop(['G1', 'G2', 'G3'], axis=1)
print(pisang.head())
#%%

#%%
pisang = pd.get_dummies(pisang, columns=['sex', 'school', 'address','famsize', 'Pstatus', 'Mjob', 'Fjob','reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])  #Mongkonversi kategori variabel menjadi variabel indikator
kuda.head()
#mengambil baris pertama dari cakue
#%%
