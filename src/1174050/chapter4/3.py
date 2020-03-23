# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:49:14 2020

@author: User
"""


#melakukan import pandas untuk membaca file csv
import pandas as pd
data_komen=pd.read_csv("D:\SEMESTER 6\Python-Artificial-Intelligence-Projects-for-Beginners-master\Chapter03\Youtube04-Eminem.csv")
#mengelompokkan komentar spam dan bukan spam
spam=data_komen.query('CLASS == 1')
nospam=data_komen.query('CLASS == 0')
#  memanggil lib vektorisasi
#melakukan fungsi bag of word dengan cara menghitung semua kata
#yang terdapat dalan file
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
# memilih colom CONTENT untuk dilakukan vektorisasi
#melakukan bag of word pada dataframe pada colom CONTENT
data_vektorisasi = vectorizer.fit_transform(data_komen['CONTENT'])
# melihat isi vektorisasi
data_vektorisasi
# melihat isi data pada baris ke 349
print(data_komen['CONTENT'][349])
#  melihat daftar kata yang di vektorisasi
#feature_names merupakan digunakan untuk mengambil nama
#kolomnya ada apa saja
dk=vectorizer.get_feature_names()
# akan melakukan randomisasi pada database nya supaya
#sempurna saat melakukan klasifikasi
acak_acak = data_komen.sample(frac=1)
# membuat data traning dan testing
dk_train=acak_acak[:300]
dk_test=acak_acak[300:]
# melakukan training pada data training dan di vektorisasi
dk_train_att=vectorizer.fit_transform(dk_train['CONTENT'])
print(dk_train_att)
#melakukan testing pada data testing dan di vektorisasi
dk_test_att=vectorizer.transform(dk_test['CONTENT'])
print(dk_test_att)
# Dimana akan mengambil label spam dan bukan spam
dk_train_label=dk_train['CLASS']
print(dk_train_label)
dk_test_label=dk_test['CLASS']
print(dk_test_label)