from sklearn import svm
#pada baris ini merupakan sebuah perintah untuk mengimport class svm dari packaged sklearn
clf = svm.SVC(gamma=0.001, C=100.)
#pada baris kedua ini clf sebagai estimator/parameter, svm.SVC sebagai class, gamma sebagai parameter untuk menetapkan nilai secara manual
clf.fit(digits.data[:-1], digits.target[:-1])
#pada baris ketiga ini clf sebagai estimator/parameter, fit sebagai metode, digits.data sebagai item, [:-1] sebagai syntax pythonnya dan menampilkan outputannya
clf.predict(digits.data[-1:])
#pada baris terakhir ini clf sebagai estimator/parameter, predict sebagai metode lainnya, digits.data sebagai item dan menampilkan outputannya