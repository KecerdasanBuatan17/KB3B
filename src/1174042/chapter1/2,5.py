#Type Casting
from sklearn import svm
#pada baris ini merupakan sebuah perintah untuk mengimport class svm dari packaged sklearn
from sklearn import random_projection
#pada baris ini merupakan sebuah perintah untuk mengimport class random projection dari packaged sklearn
rng = np.random.RandomState(0)
#rng sebagai estimator/parameter dengan nilai suatu itemnya yaitu np.random.RandomState(0)
X = rng.rand(10, 2000)
#X sebagai estimator/parameter dengan nilai item rng.rand
X = np.array(X, dtype='float32')
#X sebagai estimator/parameter dengan nilai item np.array
X.dtype
#X.dtype sebagai item pemanggil
transformer = random_projection.GaussianRandomProjection()
#transformer sebagai estimator/parameter dengan memanggil class random projection
X_new = transformer.fit_transform(X)
#X new di sini sebagai estomator/parameter dan menggunakan metode fit
X_new.dtype
#X new.dtype sebagai item


from sklearn import datasets
#pada baris ini merupakan sebuah perintah untuk mengimport class datasets dari packaged sklearn
from sklearn.svm import SVC
#pada baris ini merupakan sebuah perintah untuk mengimport class SVC dari packaged sklearn.svm
iris = datasets.load_iris()
#iris sebagai estimator/parameter dengan item datasets.load iris()
clf = SVC(gamma='scale')
#clf sebagai estimator/parameter dengan nilai class SVC pada parameter gamma sebagai set penilaian
clf.fit(iris.data, iris.target)
#estimator/parameter clf menggunakan metode fit dengan itemnya
list(clf.predict(iris.data[:3]))
#menambahkan item list dengan metode predict
clf.fit(iris.data, iris.target_names[iris.target])
#estimator/parameter clf menggunakan metode fit dengan itemnya
list(clf.predict(iris.data[:3]))
#menambahkan item list dengan metode predict



#Refitting and Updating Parameters
import numpy as np
#pada baris ini merupakan sebuah perintah untuk mengimport class svm dari np
from sklearn.svm import SVC
#pada baris ini merupakan sebuah perintah untuk mengimport class SVC dari packaged sklearn.svm
rng = np.random.RandomState(0)
#rng sebagai estimator/parameter dengan nilai suatu itemnya yaitu np.random.RandomState(0)
X = rng.rand(100, 10)
#X sebagai estimator/parameter dengan nilai item rng.rand
y = rng.binomial(1, 0.5, 100)
#y sebagai estimator/parameter dengan nilai item rng.binomial
X_test = rng.rand(5, 10)
#X test sebagai estimator/parameter dengan nilai item rng.rand
clf = SVC()
#clf sebagai estimator/parameter dan class SVC
clf.set_params(kernel='linear').fit(X, y)
#set params sebagai item
clf.predict(X_test)
#menggunakan metode predict
clf.set_params(kernel='rbf', gamma='scale').fit(X, y)
clf.predict(X_test)


#Multiclass vs. Multilabel Fitting
from sklearn.svm import SVC
#pada baris ini merupakan sebuah perintah untuk mengimport class SVC dari packaged sklearn.svm
from sklearn.multiclass import OneVsRestClassifier
#pada baris ini merupakan sebuah perintah untuk mengimport class OneVsRestClassifier dari packaged sklearn.multiclass
from sklearn.preprocessing import LabelBinarizer
#pada baris ini merupakan sebuah perintah untuk mengimport class LabelBinarizer dari packaged sklearn.preprocessing
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]
classif = OneVsRestClassifier(estimator=SVC(gamma='scale',random_state=0))
classif.fit(X, y).predict(X)
y = LabelBinarizer().fit_transform(y)
classif.fit(X, y).predict(X)


from sklearn.preprocessing import MultiLabelBinarizer
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)
classif.fit(X, y).predict(X)