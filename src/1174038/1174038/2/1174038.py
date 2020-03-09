    # -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:27:22 2020

@author: Teddy
"""

print(1174038%3)
#%% 1.Load Dataset
import pandas as pd
sirsak = pd.read_csv('student-mat.csv',sep=';')
len(sirsak)

#%% 2.generate binary label (pass/fail) based on G1+G2+G3
sirsak['pass'] = sirsak.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>= 35 else 0, axis=1)
sirsak = sirsak.drop(['G1','G2','G3'],axis=1)
sirsak.head()

#%% 3.use one-hot encoding on categorical columns
sirsak = pd.get_dummies(sirsak,columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
sirsak.head()

#%% 4.shuffle rows
sirsak = sirsak.sample(frac=1)
sirsak_train = sirsak[:500]
sirsak_test = sirsak[500:]
sirsak_train_att = sirsak_train.drop(['pass'],axis=1)
sirsak_train_pass = sirsak_train['pass']
sirsak_test_att = sirsak_test.drop(['pass'],axis=1)
sirsak_test_pass = sirsak_test['pass']
sirsak_att = sirsak.drop(['pass'],axis=1)
sirsak_pass = sirsak['pass']

import numpy as np
print("Passing: %d out %d (%.2f%%)" %(np.sum(sirsak_pass),len(sirsak_pass),100*float(np.sum(sirsak_pass))/len(sirsak_pass)))
#%% 5.fit a decision tree
from sklearn import tree
bangkoang = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)
bangkoang = bangkoang.fit(sirsak_train_att,sirsak_train_pass)

#%% 6.visualize tree
import graphviz
mangga = tree.export_graphviz(bangkoang,out_file=None,label ="all",impurity=False,proportion=True,feature_names=list(sirsak_train_att),class_names=["fail","pass"],filled=True,rounded=True)
anggur = graphviz.Source(mangga)
anggur

#%% 7.save tree
tree.export_graphviz(bangkoang,out_file="student-performance.dot",label ="all",impurity=False,proportion=True,feature_names=list(sirsak_train_att),class_names=["fail","pass"],filled=True,rounded=True)

#%% 8
bangkoang.score(sirsak_test_att,sirsak_test_pass)

#%% 9
from sklearn.model_selection import cross_val_score
belimbing = cross_val_score(bangkoang,sirsak_att,sirsak_pass,cv=5)
print("Accuracy : %0.2f (+/- %0.2f)" % (belimbing.mean(),belimbing.std() * 2))

#%% 10
for tomat in range(1,20):
   bangkoang = tree.DecisionTreeClassifier(criterion="entropy",max_depth=tomat)
    belimbing = cross_val_score(bangkoang,sirsak_att,sirsak_pass,cv=5)
    print("Max depth : %d, Accuracy : %0.2f (+/- %0.2f)" %(tomat,belimbing.mean(),belimbing.std() * 2))

#%% 11
bluberi = np.empty((19,3),float)
semangka = 0
for tomat in range(1,20):
    bangkoang = tree.DecisionTreeClassifier(criterion="entropy",max_depth=tomat)
    belimbing = cross_val_score(bangkoang,sirsak_att,sirsak_pass,cv=5)
    bluberi[semangka,0] = tomat
    bluberi[semangka,1] = belimbing.mean()
    bluberi[semangka,2] = belimbing.std() * 2
    semangka += 1
    bluberi

#%% 12
import matplotlib.pyplot as plt
rambutan, alpukat = plt.subplots()
alpukat.errorbar(bluberi[:,0],bluberi[:,1],yerr=bluberi[:,2])
plt.show()