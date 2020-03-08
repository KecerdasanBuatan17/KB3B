# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:39:34 2020

@author: lenovo
"""
#%%
# load dataset (student mat pakenya)
import pandas as pd
jeruk = pd.read_csv('D:\student-mat.csv', sep=';')
print(len(jeruk))
#%%

#%%
jeruk['pass'] = jeruk.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3'])
>= 35 else 0, axis=1)
jeruk = jeruk.drop(['G1', 'G2', 'G3'], axis=1)
print(jeruk.head())
#%%

#%%
jeruk = pd.get_dummies(jeruk, columns=['sex', 'school', 'address','famsize', 'Pstatus', 'Mjob', 'Fjob','reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])  #Mongkonversi kategori variabel menjadi variabel indikator
jeruk.head()#mengambil baris pertama dari cakue
#%%

#%%
# shuffle rows
jeruk = jeruk.sample(frac=1)
# split training and testing data
jeruk_train = jeruk[:500]
jeruk_test = jeruk[500:]
jeruk_train_att = jeruk_train.drop(['pass'], axis=1)
jeruk_train_pass = jeruk_train['pass']
jeruk_test_att = jeruk_test.drop(['pass'], axis=1)
jeruk_test_pass = jeruk_test['pass']
jeruk_att = jeruk.drop(['pass'], axis=1)
jeruk_pass = jeruk['pass']
# number of passing students in whole dataset:
import numpy as np
print("Passing: %d out of %d (%.2f%%)" % (np.sum(jeruk_pass), len(jeruk_pass),100*float(np.sum(jeruk_pass)) / len(jeruk_pass)))
#%%

#%%
# fit a decision tree
from sklearn import tree
anggur = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
anggur = anggur.fit(jeruk_train_att, jeruk_train_pass)
#%%

#%%
# visualize tree
import graphviz
dot_data = tree.export_graphviz(anggur, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(jeruk_train_att), class_names=["fail", "pass"],
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph
#%%

#%%
# save tree
tree.export_graphviz(anggur, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(jeruk_train_att), class_names=["fail", "pass"], filled=True, rounded=True)
#%%

#%%
anggur.score(jeruk_test_att, jeruk_test_pass)
#%%

#%%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(anggur,  jeruk_att, jeruk_pass, cv=5)
# show average score and +/- two standard deviations away
#(covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#%%

#%%
for max_depth in range(1, 100):
    anggur = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(anggur, jeruk_att, jeruk_pass, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std() * 2))
#%%
    
#%%
depth_acc = np.empty((19,3), float)
i = 0
for max_depth in range(1, 20):
    anggur = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(anggur, jeruk_att, jeruk_pass, cv=5)
    depth_acc[i,0] = max_depth
    depth_acc[i,1] = scores.mean()
    depth_acc[i,2] = scores.std() * 2
    i += 1

print(depth_acc)
#%%

#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2])
plt.show()
#%%