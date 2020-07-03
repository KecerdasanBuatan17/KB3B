# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 23:50:38 2020

@author: Alit
"""

# In[0]
1174057 % 3 #Hasilnya 1 maka akan menggunakan nama Kota

# In[1]
import pandas as pd #Import library pandas menggantinya nama yang akan dipanggil jadi pd
dt = pd.read_csv('C:/Users/Alit/Desktop/KB3B/src/1174057/chapter2/dataset/student-mat.csv', sep=';') #Membuat variable dt yang isinya memanggil fungsi membaca file csv
len(dt) #Menghitung jumlah data yang ada pada csv yang tadi sudah dibaca

# In[2]
# generate binary label (pass/fail) based on G1+G2+G3 (test grades, each 0-20 pts); threshold for passing is sum>=30
dt['pass'] = dt.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1) #Membuat label binary (pass/fail) berdasarkan G1+G2+G3 (testgrade, semuanya 0-20 point); Batas untuk pass adalah sum>=30
dt = dt.drop(['G1', 'G2', 'G3'], axis=1) #Meghilangkan data G1 G2 dan G3
dt.head() #Menampilkan data

# In[3]:
# use one-hot encoding on categorical columns
dt = pd.get_dummies(dt, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                               'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                               'nursery', 'higher', 'internet', 'romantic'])
dt.head()

# In[4]:
# shuffle rows
dt = dt.sample(frac=1) #Mengambil data sample dari dt
# split training and testing data
dt_train = dt[:500] #Membagi data untuk training
dt_test = dt[500:] #Membagi data untuk test

dt_train_att = dt_train.drop(['pass'], axis=1) #Meghapus data yang telah pass dan memasukkannya
dt_train_pass = dt_train['pass'] #Mengambil data yang pass saja

dt_test_att = dt_test.drop(['pass'], axis=1) #Meghapus data yang telah pass dan memasukkannya
dt_test_pass = dt_test['pass'] #Mengambil data yang pass saja

dt_att = dt.drop(['pass'], axis=1) #Meghapus data yang telah pass dan memasukkannya
dt_pass = dt['pass'] #Mengambil data yang pass saja

# number of passing students in whole dataset:
import numpy as np #Mengimport library numpy sebagai np
print("Passing: %d out of %d (%.2f%%)" % (np.sum(dt_pass), len(dt_pass), 100*float(np.sum(dt_pass)) / len(dt_pass))) #Menampilkan data

# In[5]:
# fit a decision tree
from sklearn import tree #import Decision tree dari library sklearn
df = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5) #Membuat decition tree dengan maximal depthnya 5 
df = df.fit(dt_train_att, dt_train_pass) #Memasukkan data yang akan dijadikan decition treenya


# In[6]:
# visualize tree
import graphviz #Mengimport Library Grapthviz untuk memvisualisasikan decision tree
dot_data = tree.export_graphviz(df, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(dt_train_att), class_names=["fail", "pass"], 
                                filled=True, rounded=True) #Mendefinisikan dot_data yang isikan akan berisikan data yang akan dijadikan gambar
graph = graphviz.Source(dot_data) #Memasukkan data tadi menjadi sebuah graph
graph #Menampilkan graph menggunakan graphviz


# In[7]:
# save tree
tree.export_graphviz(df, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(dt_train_att), class_names=["fail", "pass"], 
                     filled=True, rounded=True) #Digunakan untuk mengexport graph tree tadi yang telah kita buat


# In[8]:
df.score(dt_test_att, dt_test_pass) #Menghitung prediksi nilai yang akan datang dimasa depan


# In[9]:
from sklearn.model_selection import cross_val_score #Mengimport fungsi cross_val_score dari library sklearn
dl = cross_val_score(df, dt_att, dt_pass, cv=5) #Mendefinisikan dl yang isinya pembagian data menjadi 5
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (dl.mean(), dl.std() * 2)) #Menampilkan data nilai dan +/- dari dua standar deviasi


# In[10]:
for max_depth in range(1, 20): #Pengulangan menunjukkan seberapa dalam tree itu
    df = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth) #Membuat decision Tree
    dl = cross_val_score(df, dt_att, dt_pass, cv=5) #Mendefinisikan dl yang isinya pembagian data menjadi 5
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, dl.mean(), dl.std() * 2)) #Menampilkan data nilai dan +/- dari dua standar deviasi


# In[11]:
depth_acc = np.empty((19,3), float) #Membuat array baru
i = 0 #Membuat variable berisikan 0
for max_depth in range(1, 20): #Perulangan untuk memasukkan data 
    df = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)#Membuat decision Tree
    dl = cross_val_score(df, dt_att, dt_pass, cv=5) #Mendefinisikan dl yang isinya pembagian data menjadi 5
    depth_acc[i,0] = max_depth #Memasukkan data max_depth ke array depth_acc
    depth_acc[i,1] = dl.mean() #Memasukkan data rata-rata dari dl ke array depth_acc
    depth_acc[i,2] = dl.std() * 2 #Memasukkan data akar 2 dari dl ke array depth_acc
    i += 1
    
depth_acc


# In[12]:
import matplotlib.pyplot as plt #Menimport fungsi pyplot dari library matplotlib sebagai plt 
fig, ax = plt.subplots() #Membuat plot baru
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2]) #Mengisikan data plot
plt.show() #Menampilkan plot 
