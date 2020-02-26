from sklearn import svm
#pada baris ini merupakan sebuah perintah untuk mengimport class svm dari packaged sklearn
from sklearn import datasets
#pada baris ini merupakan sebuah perintah untuk mengimport class datasets dari packaged sklearn
clf = svm.SVC(gamma='scale')
#pada baris ketga ini clf sebagai estimator/parameter, svm.SVC sebagai class, gamma sebagai parameter untuk menetapkan nilai secara manual dengan nilai scale
iris = datasets.load_iris()
#pada baris keempat ini iris sebagai estimator/parameter, datasets.load iris() sebagai item dari suatu nilai
X, y = iris.data, iris.target
#pada baris kelima ini X, y sebagai estimator/parameter, iris.data, iris.target sebagai item dari 2 nilai yang ada
clf.fit(X, y)
#pada baris keenam ini clf sebagai estimator/parameter dengan menggunakan metode fit untuk memanggil estimator X, y dengan outputannya


import pickle
#pickle merupakaan sebuah class yang di import
s = pickle.dumps(clf)
#pada baris ini s sebagai estimator/parameter dengan pickle.dumps merupakan suatu nilai/item dari estimator/parameter clf
clf2 = pickle.loads(s)
#pada baris ini clf2 sebagai estimator/parameter, pickle.loads sebagai suatu item, dan s sebagai estimator/parameter yang dipanggil
clf2.predict(X[0:1])
#pada baris ini clf2.predict sebagai suatu item dengan menggunakan metode predict untuk menentukkan suatu nilai dari (X[0:1])
y[0]
#pada estimator/parameter y berapapun angka yang diganti nilainya akan selalu konstan yaitu 0


from joblib import dump, load
#pada baris berikut ini merupakan sebuah perintah untuk mengimport class dump, load dari packaged joblib
dump(clf, 'filename.joblib')
#pada baris berikutnya dump di sini sebagai class yang didalamnya terdapat nilai dari suatu item clf dan data joblib
clf = load('filename.joblib')
#pada baris terakhir clf sebagai estimato/parameter dengan suatu nilai load berfungsi untuk mengulang data sebelumnya
