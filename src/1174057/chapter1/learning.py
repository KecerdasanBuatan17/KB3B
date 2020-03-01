from sklearn import svm # perintah untuk mengimport class svm dari packaged sklearn

digits = datasets.load_digits() #memuat dan memasukkan dataset digits ke variabel digits

clf = svm.SVC(gamma=0.001, C=100.) #clf sebagai estimator/parameter , svm.SVC sebagai class , gamma sebagai parameter untuk menetapkan nilai secara manual

clf.fit(digits.data[:-1], digits.target[:-1]) # clf sebagai estimator/parameter , f i t sebagai metode , digits . data sebagai item , [:1] sebagai syntax pythonnya dan menampilkan outputannya 

print(clf.predict(digits.data[-1:])) #clf sebagai estimator/parameter , predict sebagai metode lainnya , digits . data sebagai item dan menampilkan outputannya
