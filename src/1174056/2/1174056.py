    # -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:27:22 2020

@author: Rangga
"""

print(1174056%3)
#%% 1.Load Dataset
import pandas as pd
pisang = pd.read_csv('student-mat.csv',sep=';')
len(pisang)

#%% 2.generate binary label (pass/fail) based on G1+G2+G3
pisang['pass'] = pisang.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>= 35 else 0, axis=1)
pisang = pisang.drop(['G1','G2','G3'],axis=1)
pisang.head()

#%% 3.use one-hot encoding on categorical columns
pisang = pd.get_dummies(pisang,columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
pisang.head()

#%% 4.shuffle rows
pisang = pisang.sample(frac=1)
pisang_train = pisang[:500]
pisang_test = pisang[500:]
pisang_train_att = pisang_train.drop(['pass'],axis=1)
pisang_train_pass = pisang_train['pass']
pisang_test_att = pisang_test.drop(['pass'],axis=1)
pisang_test_pass = pisang_test['pass']
pisang_att = pisang.drop(['pass'],axis=1)
pisang_pass = pisang['pass']

import numpy as np
print("Passing: %d out %d (%.2f%%)" %(np.sum(pisang_pass),len(pisang_pass),100*float(np.sum(pisang_pass))/len(pisang_pass)))
#%% 5.fit a decision tree
from sklearn import tree
apel = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)
apel = apel.fit(pisang_train_att,pisang_train_pass)

#%% 6.visualize tree
import graphviz
mangga = tree.export_graphviz(apel,out_file=None,label ="all",impurity=False,proportion=True,feature_names=list(pisang_train_att),class_names=["fail","pass"],filled=True,rounded=True)
anggur = graphviz.Source(mangga)
anggur

#%% 7.save tree
tree.export_graphviz(apel,out_file="student-performance.dot",label ="all",impurity=False,proportion=True,feature_names=list(pisang_train_att),class_names=["fail","pass"],filled=True,rounded=True)

#%% 8
apel.score(pisang_test_att,pisang_test_pass)

#%% 9
from sklearn.model_selection import cross_val_score
belimbing = cross_val_score(apel,pisang_att,pisang_pass,cv=5)
print("Accuracy : %0.2f (+/- %0.2f)" % (belimbing.mean(),belimbing.std() * 2))

#%% 10
for tomat in range(1,20):
    apel = tree.DecisionTreeClassifier(criterion="entropy",max_depth=tomat)
    belimbing = cross_val_score(apel,pisang_att,pisang_pass,cv=5)
    print("Max depth : %d, Accuracy : %0.2f (+/- %0.2f)" %(tomat,belimbing.mean(),belimbing.std() * 2))

#%% 11
bluberi = np.empty((19,3),float)
semangka = 0
for tomat in range(1,20):
    apel = tree.DecisionTreeClassifier(criterion="entropy",max_depth=tomat)
    belimbing = cross_val_score(apel,pisang_att,pisang_pass,cv=5)
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