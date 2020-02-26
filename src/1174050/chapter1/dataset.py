from sklearn import datasets
#perintah untuk mengimport class datasets dari packaged sklearn
iris = datasets.load_iris()
#iris merupakan suatu estimator/parameter yang berfungsi untuk mengambil data pada item datasets.load iris
digits = datasets.load_digits()
#digits merupakan suatu estimator/parameter yang berfungsi untuk mengambil data pada item datasets.load digits
print(digits.data)
#perintah yang berfungsi untuk menampilkan estimator/parameter yang dipanggil pada item digits.data dan menampilkan outputannya
digits.target
#untuk mengambil target pada estimator/parameter digits dan menampilkan outputannya
digits.images[0]
#untuk mengambil images[0] pada estimator/parameter digits dan menampilkan outputannyal