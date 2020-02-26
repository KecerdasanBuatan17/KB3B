# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:06:05 2020

@author: User
"""

import pickle
#pickle merupakaan sebuah class yang di import
s = pickle.dumps(clf)
#s sebagai estimator/parameter dengan pickle.dumps merupakan suatu nilai/item dari estimator/parameter clf
clf2 = pickle.loads(s)
#clf2 sebagai estimator/parameter, pickle.loads sebagai suatu item, dan s sebagai estimator/parameter yang dipanggil
clf2.predict(X[0:1])
#clf2.predict sebagai suatu item dengan menggunakan metode predict untuk menentukkan suatu nilai dari (X[0:1])
y[0]
#pada estimator/parameter y berapapun angka yang diganti nilainya akan selalu konstan yaitu 0


from joblib import dump, load
#sebuah perintah untuk mengimport class dump, load dari packaged joblib
dump(clf, 'filename.joblib')
#dump di sini sebagai class yang didalamnya terdapat nilai dari suatu item clf dan data joblib
clf = load('filename.joblib')
#clf sebagai estimato/parameter dengan suatu nilai load berfungsi untuk mengulang data sebelumnya