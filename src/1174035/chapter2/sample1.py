# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:06:57 2020

@author: TIKOMDIK 01
"""
#%%
cakue = 1174035%3 #Inisiasi variable cakue, untuk cek NPM mod 3 hasilnya berapa
print(cakue) #Menampil
#%%
import pandas as pd #mengimpor library pandas
cakue = pd.read_csv('student-mat.csv', sep=';') #data csv dibaca lalu dipisahkan dengan titik koma';'
len(cakue) #menghitung total nilai(panjang kumpulan nilai/array) yang terpisahkan dari csv tersebut
#%%
cakue['pass'] = cakue.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1) 
cakue = cakue.drop(['G1', 'G2', 'G3'], axis=1) 
cakue.head()#mengambil baris pertama dari cakue
#%%
cakue = pd.get_dummies(cakue, columns=['sex', 'school', 'address','famsize', 'Pstatus', 'Mjob', 'Fjob','reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])  #Mongkonversi kategori variabel menjadi variabel indikator
cakue.head()#mengambil baris pertama dari cakue
#%%
# shuffle rows 
cakue = cakue.sample(frac=1) #mengenerate sampel acak dari baris atau kolom dari cakue
# split training and testing data 
cakue_train = cakue[:500]#Mengambil data array dengan batas index 500
cakue_test = cakue[500:] #Mengambil data array dengan awal index 500
cakue_train_att = cakue_train.drop(['pass'], axis=1) #
cakue_train_pass = cakue_train['pass'] #Mengambil data cakue_train dengan index object 'pass' dan dimasukkan ke variable cakue_train_pass
cakue_test_att = cakue_test.drop(['pass'], axis=1)  
cakue_test_pass = cakue_test['pass'] #Mengambil data cakue_train dengan index object 'pass' dan dimasukkan ke variable cakue_test_pass
cakue_att = cakue.drop(['pass'], axis=1)  
cakue_pass = cakue['pass']#Mengambil data cakue_train dengan index object 'pass' dan dimasukkan ke variable cakue_pass
# number of passing students in whole dataset: 
import numpy as np #mengimport library numpy
print("Passing: %d out of %d (%.2f%%)" % (np.sum(cakue_pass), len(cakue_pass), 100*float(np.sum(cakue_pass)) / len(cakue_pass))) #Menampilkan hasil integer dan float untuk melihat passing datanya
#%%
from sklearn import tree #Mengimport class tree dari library sklearn
kwetiau = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5) #Memanggil fungsi untuk melakukan prediksi dengan aturan keputusan yang simpel
kwetiau = kwetiau.fit(cakue_train_att, cakue_train_pass) #Untuk melakukan pengiriman data training set ke method fit
#%%
import graphviz #mengimport class graphviz
dot_data = tree.export_graphviz(kwetiau, out_file=None, label="all", impurity=False, proportion=True,feature_names=list(cakue_train_att),class_names=["fail", "pass"],filled=True, rounded=True) #Untuk mengeksport data ke format graphfiz
graph = graphviz.Source(dot_data)  #Untuk mengambil sumber data berformat graphviz dan dimasukkan ke variable graph
graph #Menampilkan isi variable graph untuk di spyder
#%%
tree.export_graphviz(kwetiau, out_file="student-performance.dot", label="all", impurity=False, proportion=True, feature_names=list(cakue_train_att), class_names=["fail", "pass"],filled=True, rounded=True) #Untuk mengeksport data ke format graphfiz
#%%
kwetiau.score(cakue_test_att, cakue_test_pass) #Untuk melakukan penilaian sesuai dengan drop - drop pada variable cakue_test_att dan cakue_test_pass
#%%
from sklearn.model_selection import cross_val_score  #Mengambil class cross_val_score dari library sklearn.model_selection
scores = cross_val_score(kwetiau, cakue_att, cakue_pass, cv=5)  #Untuk mengevaluasi nilai menggunakan metode cross-validation
# show average score and +/- two standard deviations away 
#(covering 95% of scores) 
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) #Menampilkan score
#%%
for max_depth in range(1, 20): #Looping dengan for berdasarkan nilai dari 1 sampai 20 dengan variable index yaitu max_depth
    kwetiau = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)  #Memanggil fungsi untuk melakukan prediksi dengan aturan keputusan yang simpel
    scores = cross_val_score(kwetiau, cakue_att, cakue_pass, cv=5) #Untuk mengevaluasi nilai menggunakan metode cross-validation
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std() * 2) )#Menampilkan score
#%%
depth_acc = np.empty((19,3), float)  #Mengembalikan array dengan format bentuk dan tipe
i = 0#Inisiasi variable
for max_depth in range(1, 20): #Looping dengan for berdasarkan nilai dari 1 sampai 20 dengan variable index yaitu max_depth
    kwetiau = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth) #Memanggil fungsi untuk melakukan prediksi dengan aturan keputusan yang simpel
    scores = cross_val_score(kwetiau, cakue_att, cakue_pass, cv=5)  #Untuk mengevaluasi nilai menggunakan metode cross-validation
    depth_acc[i,0] = max_depth 
    depth_acc[i,1] = scores.mean()
    depth_acc[i,2] = scores.std() * 2 
    i += 1
depth_acc
#%%
import matplotlib.pyplot as plt #Mengimport library matplotlib>pyplot
fig, ax = plt.subplots() #Menampilkan grafik dari library matplotlib
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2]) 
plt.show()
