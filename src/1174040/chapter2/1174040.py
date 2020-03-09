# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:39:04 2020

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
pisang.head()
#mengambil baris pertama dari cakue
#%%

#%%
# shuffle rows
pisang = pisang.sample(frac=1)
# split training and testing data
pisang_train = pisang[:500]
pisang_test = pisang[500:]
pisang_train_att = pisang_train.drop(['pass'], axis=1)
pisang_train_pass = pisang_train['pass']
pisang_test_att = pisang_test.drop(['pass'], axis=1)
pisang_test_pass = pisang_test['pass']
pisang_att = pisang.drop(['pass'], axis=1)
pisang_pass = pisang['pass']
# number of passing students in whole dataset:
import numpy as np
print("Passing: %d out of %d (%.2f%%)" % (np.sum(pisang_pass), len(pisang_pass),100*float(np.sum(pisang_pass)) / len(pisang_pass)))
#%%

#%%
# fit a decision tree
from sklearn import tree
apel = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
apel = apel.fit(pisang_train_att, pisang_train_pass)
print(apel)
#%%

#%%
# visualize tree
import graphviz
dot_data = tree.export_graphviz(apel, out_file=None, label="all", impurity=False, proportion=True, feature_names=list(pisang_train_att), class_names=["fail", "pass"], filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph
#%%

#%%
# save tree
tree.export_graphviz(apel, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(pisang_train_att), class_names=["fail", "pass"], filled=True, rounded=True)
#%%

#%%
apel.score(pisang_test_att, pisang_test_pass)
#%%

#%%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(apel,  pisang_att, pisang_pass, cv=5)
# show average score and +/- two standard deviations away
#(covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#%%

#%%
for max_depth in range(1, 100):
    apel = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(apel, pisang_att, pisang_pass, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std() * 2))
#%%
    
#%%
depth_acc = np.empty((19,3), float)
i = 0
for max_depth in range(1, 20):
    apel = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(apel, pisang_att, pisang_pass, cv=5)
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